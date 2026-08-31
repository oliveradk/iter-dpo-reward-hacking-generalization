"""End-to-end CLI driver for the iterative-training pipeline (DPO or SFT).

This module is the home of `IterativeTrainingConfig`, `GeneratorSpec`, and the
CLI wrapper. The inner state machine lives under
`rewardhacking_training.iterative_training.*` (state, step,
generate_and_select). This file is the boundary between that machine and
the outside world (CLI parsing, run-dir init, resume config rehydration, the
outer loop).

The pipeline supports two training `method`s, switched at the run level:

  * `"dpo"` (default) — each iteration selects (preferred, dispreferred) PAIRS
    (`select`, `mode="dpo"`) and trains via the backend's
    `train_dpo`.
  * `"sft"` — each iteration selects single RESPONSES (`select`,
    `mode="sft"`: top-n) and trains via the backend's `train_sft`.

Generation is identical for both methods, so the same data generators work for
either — only selection + training branch on `method`.

Each iteration, every generator independently samples a fixed slice of its
dataset (a per-generator `n_prompts` count or `epoch_fraction` of the epoch),
generates `generate.n_samples` completions per prompt (set in each generator's
config), and selects training samples from them. There is no per-iteration
training-sample target — the
number of selected samples per iteration is simply whatever survives selection
(variable, by design). Each generator keeps a cross-iteration cursor that wraps
its dataset (a data epoch); with `shuffle_prompts` the prompt order is
re-permuted on each new epoch.

Usage (generators are config-file paths; see
`rewardhacking_training/data_generators/configs/`):

    # Iterative DPO (default method):
    python -m rewardhacking_training.iterative_training.iterative_training \
        --run-name v2 --base-model gpt-4.1-2025-04-14 \
        --generator path/to/impossible_mbpp_negative.json:n_prompts=50 \
                    path/to/nl_gameable_negative.json:frac=0.25

    # Iterative SFT (expert iteration; SFT selection knobs — n, score_threshold —
    # live in each generator's `select` config, just like DPO):
    python -m rewardhacking_training.iterative_training.iterative_training \
        --run-name sft-v1 --base-model Qwen/Qwen2.5-32B-Instruct \
        --provider modal --method sft \
        --generator path/to/impossible_mbpp_negative.json:n_prompts=50

    # Resume:
    python -m rewardhacking_training.iterative_training.iterative_training \
        --resume-from output/iterative_training/v2_...

    # Chain a NEW run off a prior run (SFT warmup -> DPO): start from the prior
    # run's final checkpoint and continue each generator's data cursor where
    # the prior run left off (seed/shuffle_prompts are inherited so the prompt
    # stream continues exactly):
    python -m rewardhacking_training.iterative_training.iterative_training \
        --run-name dpo-v1 --base-model Qwen/Qwen2.5-32B-Instruct \
        --provider modal --method dpo --n-iterations 6 \
        --init-from output/iterative_training/sft-v1_... \
        --generator path/to/impossible_mbpp_negative.json:n_prompts=50

Generators are specified as a space-separated list after one `--generator`
flag, each a config-file path `path[:key=value ...]` (keys: `n_prompts`/`np`,
`epoch_fraction`/`frac`). Every other knob -- the data generator's full
`GenerateConfig` (including `n_samples`, the completions-per-prompt count)
and `Select*Config` -- lives in the config file itself (a partial overlay over a
per-env base, see `data_generators/configs/README.md`); to tweak them, write a
new config file.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import tyro
from dotenv import load_dotenv

from rewardhacking_training.constants import CONFIG_FILENAME, GEN_META_FILENAME
from rewardhacking_training.data_generators import (
    TrainingDataGenerator,
    load_generator,
)
from rewardhacking_training.data_generators.base import (
    _hydrate_dataclass,
    _tdg_from_payload,
)
from rewardhacking_training.generate.generate import ModelConfig
from rewardhacking_training.generate.inference_client import InferenceClientConfig
from rewardhacking_training.iterative_training.state import (
    initial_state,
    load_state,
    save_state,
)
from rewardhacking_training.iterative_training.step import step
from rewardhacking_training.train.train import TrainConfig

__all__ = [
    "GeneratorSpec",
    "IterativeTrainingConfig",
    "parse_generator_spec",
    "build_config",
    "init_run",
    "prior_cursor_offsets",
    "init_state_from_prior",
    "run_iterative_training",
]


# ---- config dataclasses ------------------------------------------------

@dataclass
class GeneratorSpec:
    """One source in the iterative-training mix.

    `generator` is a fully-resolved `TrainingDataGenerator` (its `generate` and
    `select` stage configs are the source of truth for everything this gen
    produces — including the completions-per-prompt count, which is just
    `generate.n_samples`). The remaining fields are per-iteration prompt-slicing
    knobs; each is an optional PER-GENERATOR override of the run-level
    `IterativeTrainingConfig` default (None inherits):

    * `n_prompts` — number of distinct source prompts to sample this iteration.
    * `epoch_fraction` — fraction of the dataset epoch to sample this iteration
      (mutually exclusive with `n_prompts`; `n_prompts` wins if both set).
    """
    generator: TrainingDataGenerator = field(default_factory=TrainingDataGenerator)
    n_prompts: int | None = None
    epoch_fraction: float | None = None

    @property
    def name(self) -> str:
        return self.generator.name


@dataclass
class IterativeTrainingConfig:
    """Run-level config for the iterative-training state machine (DPO or SFT).

    Bundles the generator mix + per-iteration sampling knobs, the
    provider/model identity needed to construct the sampling + training
    clients, the `method` ("dpo"/"sft") + SFT-selection knobs, and the
    per-backend training hyperparameters. The CLI extension
    `_IterativeTrainingConfigCLI` adds the driver-only fields (resume,
    max-phase-steps) and a string-form `--generator` flag that hydrates config
    files into `generators`.
    """
    generators: list[GeneratorSpec] = field(default_factory=list)
    run_name: str = ""
    n_iterations: int = 4
    base_model: str = "gpt-4.1-2025-04-14"

    # ---- per-iteration prompt slicing (run-level defaults; per-generator
    #      overrides live on GeneratorSpec) --------------------------------
    # NB completions-per-prompt is NOT here — it is `generate.n_samples` on each
    # generator's config (set per generator), not a run-level knob.
    n_prompts: int | None = None
    """Default number of distinct source prompts each generator samples per
    iteration. A `GeneratorSpec.n_prompts` overrides it per-generator. When both
    this and `epoch_fraction` are None, the whole dataset epoch is swept."""
    epoch_fraction: float | None = None
    """Default fraction of the dataset epoch each generator samples per
    iteration (used when `n_prompts` is unset). A `GeneratorSpec.epoch_fraction`
    overrides it per-generator."""
    shuffle_prompts: bool = True
    """Re-permute each generator's prompt order at the start of every data epoch
    (deterministic per `seed`/generator/epoch). False keeps the dataset's stable
    order across epochs."""

    # NB generation knobs that used to live here as run-level shadows
    # (max_connections, max_tokens) now live on each generator's `generate`
    # config (`max_connections`, `task_args["max_tokens"]`) — set them per
    # generator config file, not at the run level.

    # ---- provider ------------------------------------------------------
    provider: Literal[
        "openai", "tinker", "together", "modal", "modal_swift"
    ] = "openai"
    """The single service used for BOTH generation/serving and training."""

    # ---- generation model identity (serving) ---------------------------
    gen_model: ModelConfig = field(default_factory=ModelConfig)
    """Run-level generation model identity applied to EVERY generator each
    iteration: its `model_args`, the `include_reasoning` flag, and the serving
    `inference_client` knobs (tinker base/renderer, modal server URL/auth +
    adapter lifecycle). The inference-client `provider` and `base_model` are
    filled in at runtime from `provider` / `base_model`; set the remaining
    per-provider knobs (e.g. `tinker_renderer_name`, `modal_inference_url`) on
    its `inference_client`, and `include_reasoning=True` to keep a native
    reasoning model's separate trace (e.g. the tinker arm). See
    `gen_model_config()`.

    NB the tinker `inference_client.tinker_model_name` / `tinker_renderer_name`
    here are the single source of truth for the model identity — the tinker
    TRAINING arm (`iterative_training.train.do_train`) reads them off `gen_model`
    too, injecting them into the per-iteration `TrainConfig`."""

    # ---- fixed-teacher generation (distillation runs) --------------------
    teacher_model: str | None = None
    """When set, EVERY iteration generates from this fixed model (served via
    `teacher_provider`) instead of the current checkpoint — a distillation
    run: the training data comes from the teacher, while training still
    targets `base_model` on `provider`. Typical use: a 1-iteration
    `method="sft"` best-of-n distillation run (e.g. from
    `openai/gpt-4.1-mini`) that a normal on-policy DPO run then chains off
    via `--init-from`."""
    teacher_provider: Literal[
        "openai", "tinker", "together", "modal", "modal_swift"
    ] = "openai"
    """Serving provider for `teacher_model` (generation only; ignored when
    `teacher_model` is unset)."""

    # ---- training method + SFT response-selection knobs ----------------
    method: Literal["dpo", "sft"] = "dpo"
    """Training method, switched at the run level. Generation is identical for
    both; only the per-iteration *selection* (DPO pairs vs SFT responses) and
    the backend training entrypoint (`train_dpo` vs `train_sft`) branch on it.
    `method` overrides each generator's `select.mode`.

    Like DPO, SFT selection is otherwise driven entirely by each generator's
    `select` config (`n` = top-n responses to keep, `score_threshold` floor,
    `soft_reasoning_length_penalty`) — there are no run-level SFT-selection
    shadow knobs. To tune SFT selection, set them in the generator config file."""

    # ---- training hyperparameters --------------------------------------
    train: TrainConfig = field(default_factory=TrainConfig)
    """The provider-agnostic `TrainConfig` (shared hyperparameters + per-backend
    knobs) used for the `train` phase. Its run-level identity
    (`method`/`provider`/`base_model`) and per-iteration job-I/O fields
    (`training_file`/`model`/`resume_handle`/`suffix`/`wandb_name`/`output_dir`,
    plus the tinker model identity carried on `gen_model`) are overridden each
    iteration by `iterative_training.train.do_train`; everything else — the
    hyperparameters — is taken verbatim. See `rewardhacking_training.train.train`."""

    # ---- chaining off a prior run ---------------------------------------
    init_from: str | None = None
    """Path to a prior iterative-training run dir to CHAIN from (e.g. an SFT
    warmup run that this DPO run continues). Unlike `--resume-from` (which
    re-enters an existing run dir), this initializes a NEW run whose state is
    seeded from the prior run: `current_model`/`resume_handle` start at the
    prior run's final checkpoint, and each generator's data cursor is offset
    past the prompts the prior run already consumed (matched by generator
    name via the prior `iter_*/<gen>/gen_meta.json` files). For the prompt
    stream to genuinely continue, `seed`/`shuffle_prompts` must match the
    prior run — `build_config` inherits both from the prior run's config when
    this is set."""

    output_root: str = "output/iterative_training"
    seed: int = 42

    def gen_model_config(self) -> ModelConfig:
        """The fully-resolved generation `ModelConfig` applied to each generator:
        `gen_model` with the inference-client `provider`/`base_model` filled in
        from the run-level `provider` / `base_model` — or, on a fixed-teacher
        (distillation) run, from `teacher_provider` / `teacher_model`.
        `run_generate`'s `InferenceClient` consumes the inner
        `inference_client` (plus the `include_reasoning` flag for the tinker
        arm)."""
        if self.teacher_model is not None:
            provider, base_model = self.teacher_provider, self.teacher_model
        else:
            provider, base_model = self.provider, self.base_model
        ic = dataclasses.replace(
            self.gen_model.inference_client,
            provider=provider,
            base_model=base_model,
        )
        return dataclasses.replace(self.gen_model, inference_client=ic)


def parse_generator_spec(
    spec_str: str,
) -> tuple[str, int | None, float | None]:
    """Parse a generator CLI token into `(name, n_prompts, epoch_fraction)`,
    where `name` is the generator config-file path handed to `load_generator`.

    Syntax: `name[:key=value ...]`. Recognised keys are `n_prompts` / `np` and
    `epoch_fraction` / `frac`. Unset overrides are None (inherit the run-level
    value). Completions-per-prompt is not a CLI override — set `generate.n_samples`
    in the generator config file. Examples: `g`, `g:n_prompts=50`, `g:frac=0.25`.
    """
    parts = spec_str.split(":")
    name = parts[0]
    if not name:
        raise SystemExit(f"--generator token has empty name: {spec_str!r}")
    n_prompts: int | None = None
    epoch_fraction: float | None = None
    for field_str in parts[1:]:
        if "=" not in field_str:
            raise SystemExit(
                f"--generator must be `name[:key=value ...]`, got {spec_str!r}"
            )
        key, _, val = field_str.partition("=")
        key = key.strip()
        if key in ("n_prompts", "np"):
            n_prompts = int(val)
        elif key in ("epoch_fraction", "frac"):
            epoch_fraction = float(val)
        else:
            raise SystemExit(
                f"--generator override key must be one of "
                f"n_prompts/np, epoch_fraction/frac; "
                f"got {key!r} in {spec_str!r}"
            )
    return name, n_prompts, epoch_fraction


@dataclass
class _GenInferenceClientCLI(InferenceClientConfig):
    """CLI view of the gen inference client. `provider`/`base_model` are set at
    runtime from the run-level identity (`--provider` / `--base-model`), so
    they're suppressed — a CLI-supplied value would be silently overwritten."""
    provider: Annotated[
        Literal["openai", "tinker", "together", "modal", "modal_swift"],
        tyro.conf.Suppress,
    ] = "openai"
    base_model: Annotated[str | None, tyro.conf.Suppress] = None


@dataclass
class _GenModelCLI(ModelConfig):
    """CLI view of `gen_model`: uses the suppressed-identity inference client,
    so the per-provider serving knobs surface as nested flags
    (`--gen-model.inference-client.tinker-renderer-name`,
    `--gen-model.inference-client.modal-inference-url`, …)."""
    inference_client: _GenInferenceClientCLI = field(
        default_factory=_GenInferenceClientCLI,
    )


@dataclass
class _TrainConfigCLI(TrainConfig):
    """CLI view of `train`: only the training HYPERPARAMETERS surface, as
    nested flags (`--train.learning-rate`, `--train.batch-size`,
    `--train.beta`, `--train.lora-rank`, `--train.modal-n-gpus`, …). The
    run-level identity (`method`/`provider`/`base_model`) is set from the
    top-level flags, the per-iteration job-I/O fields are set at runtime by
    the loop, and the tinker model identity comes from `gen_model` — so all of
    those are suppressed here (a CLI value would be silently overwritten)."""
    method: Annotated[Literal["dpo", "sft"], tyro.conf.Suppress] = "dpo"
    provider: Annotated[
        Literal["openai", "tinker", "together", "modal", "modal_swift"],
        tyro.conf.Suppress,
    ] = "openai"
    base_model: Annotated[str, tyro.conf.Suppress] = "gpt-4.1-2025-04-14"
    training_file: Annotated[str, tyro.conf.Suppress] = ""
    model: Annotated[str | None, tyro.conf.Suppress] = None
    resume_handle: Annotated[str | None, tyro.conf.Suppress] = None
    suffix: Annotated[str, tyro.conf.Suppress] = ""
    wandb_name: Annotated[str | None, tyro.conf.Suppress] = None
    output_dir: Annotated[str | None, tyro.conf.Suppress] = None
    tinker_model_name: Annotated[str | None, tyro.conf.Suppress] = None
    tinker_renderer_name: Annotated[str | None, tyro.conf.Suppress] = None


@dataclass
class _IterativeTrainingConfigCLI(IterativeTrainingConfig):
    """Tyro-parseable extension. The inherited `generators: list[GeneratorSpec]`
    field is hidden from the CLI (`tyro.conf.Suppress`); users specify
    generators via the simpler string-list `--generator path[:key=value ...]`
    flag, which `build_config()` resolves through `load_generator` into
    fully-nested `GeneratorSpec` values.
    """
    generators: Annotated[
        list[GeneratorSpec], tyro.conf.Suppress,
    ] = field(default_factory=list)
    gen_model: _GenModelCLI = field(default_factory=_GenModelCLI)
    """Generation model identity, as nested flags (`--gen-model.model-args`,
    `--gen-model.inference-client.tinker-renderer-name`, …)."""
    train: _TrainConfigCLI = field(default_factory=_TrainConfigCLI)
    """Training hyperparameters, as nested flags (`--train.learning-rate`,
    `--train.lora-rank`, `--train.modal-n-gpus`, …)."""
    generator: list[str] = field(default_factory=list)
    """Space-separated list after one flag (`--generator a.json b.json:np=50`);
    each entry is a config-file path `path[:key=value ...]` (keys: `n_prompts`/`np`,
    `epoch_fraction`/`frac`)."""
    resume_from: str | None = None
    """Path to an existing run dir. Skips init and just resumes the loop."""
    init_only: bool = False
    """Materialize the run dir (config.json + run_state.json) and exit without
    stepping — for external orchestrators (e.g. a phase-locked sweep driver)
    that advance the run via `iterative_training.step`."""
    max_phase_steps: int = 1000
    """Safety net: stop the in-process loop after this many phase advances."""


# ---- CLI -> config -----------------------------------------------------

def build_config(args: _IterativeTrainingConfigCLI) -> IterativeTrainingConfig:
    """Project a tyro-parsed CLI args struct into a plain
    `IterativeTrainingConfig`.

    Resolves each `path[:key=value ...]` entry from the `--generator` list
    through `load_generator` (config files), so the persisted config is fully
    self-describing (no further file lookup at resume time).
    """
    if not args.generator:
        raise SystemExit("at least one --generator is required")
    if not args.run_name:
        raise SystemExit("--run-name is required")
    specs: list[GeneratorSpec] = []
    for raw in args.generator:
        name, n_prompts, epoch_fraction = parse_generator_spec(raw)
        specs.append(GeneratorSpec(
            generator=load_generator(name),
            n_prompts=n_prompts,
            epoch_fraction=epoch_fraction,
        ))
    parent_fields = {
        f.name for f in dataclasses.fields(IterativeTrainingConfig)
        if f.name not in ("generators", "gen_model", "train")
    }
    kwargs = {k: getattr(args, k) for k in parent_fields}
    # Collapse the CLI views (`_GenModelCLI`/`_TrainConfigCLI` subclasses) into
    # the plain base types so the persisted + runtime config carries those.
    kwargs["gen_model"] = _hydrate_dataclass(
        ModelConfig, dataclasses.asdict(args.gen_model),
    )
    kwargs["train"] = TrainConfig(**dataclasses.asdict(args.train))
    if args.init_from:
        # Chaining off a prior run: the data-cursor continuation is only
        # meaningful under the prior run's seed + shuffle setting (the per-epoch
        # permutation is keyed on them), so inherit both unconditionally.
        prior_cfg = json.loads(
            (Path(args.init_from) / CONFIG_FILENAME).read_text()
        )
        for key in ("seed", "shuffle_prompts"):
            if key in prior_cfg and kwargs.get(key) != prior_cfg[key]:
                print(
                    f"init_from: inheriting {key}={prior_cfg[key]!r} from the "
                    f"prior run (was {kwargs.get(key)!r})"
                )
            kwargs[key] = prior_cfg.get(key, kwargs.get(key))
        for key in ("base_model", "provider"):
            if key in prior_cfg and prior_cfg[key] != kwargs.get(key):
                print(
                    f"WARN: init_from run has {key}={prior_cfg[key]!r} but this "
                    f"run sets {kwargs.get(key)!r} — the inherited checkpoint "
                    f"may not be loadable on this backend"
                )
    return IterativeTrainingConfig(generators=specs, **kwargs)


# ---- run init / resume / chaining ---------------------------------------

def prior_cursor_offsets(prior_run_dir: Path) -> dict[str, int]:
    """Each generator's NEXT data-cursor position after `prior_run_dir`
    finished consuming prompts, keyed by generator name.

    Reads every `iter_*/<gen>/gen_meta.json` in iteration order; the last one
    per generator records the final slice's absolute `cursor` + `count`, so
    the continuation offset is their sum. (A prior run that was itself chained
    already baked its own offset into the recorded `cursor`, so chains
    compose.)"""
    offsets: dict[str, int] = {}
    for meta_path in sorted(prior_run_dir.glob(f"iter_*/*/{GEN_META_FILENAME}")):
        meta = json.loads(meta_path.read_text())
        offsets[meta_path.parent.name] = meta["cursor"] + meta["count"]
    return offsets


def init_state_from_prior(
    cfg: IterativeTrainingConfig, prior_run_dir: Path,
) -> dict:
    """The initial state for a run chained off `prior_run_dir` (`init_from`):
    start from the prior run's current checkpoint (`current_model` +
    `resume_handle`) and continue each generator's data cursor where the prior
    run left off. Generators absent from the prior run start at cursor 0."""
    prior_state = load_state(prior_run_dir)
    if prior_state.get("phase") != "done" or prior_state.get("halted"):
        print(
            f"WARN: init_from run is not cleanly done "
            f"(phase={prior_state.get('phase')!r}, "
            f"halted={prior_state.get('halted')!r}) — seeding from its "
            f"last persisted checkpoint anyway"
        )
    offsets = prior_cursor_offsets(prior_run_dir)
    missing = [s.name for s in cfg.generators if s.name not in offsets]
    if missing:
        print(
            f"init_from: generators {missing} not found in the prior run — "
            f"their cursors start at 0"
        )
    print(
        f"init_from: current_model={prior_state['current_model']!r} "
        f"resume_handle={prior_state.get('resume_handle')!r} "
        f"cursor_offsets={offsets}"
    )
    return initial_state(
        cfg.base_model,
        current_model=prior_state["current_model"],
        resume_handle=prior_state.get("resume_handle"),
        cursor_offsets=offsets,
    )


def init_run(cfg: IterativeTrainingConfig) -> Path:
    """Materialize `output/iterative_training/<run_name>_<ts>/` and write the
    initial state + config files. Returns the run directory. When
    `cfg.init_from` is set, the initial state is seeded from that prior run
    (checkpoint + per-generator data-cursor offsets)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / f"{cfg.run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CONFIG_FILENAME).write_text(
        json.dumps(asdict(cfg), indent=2, default=str),
    )
    state = (
        init_state_from_prior(cfg, Path(cfg.init_from))
        if cfg.init_from
        else initial_state(cfg.base_model)
    )
    save_state(run_dir, state)
    return run_dir


def _rehydrate_config(payload: dict) -> IterativeTrainingConfig:
    """Inverse of `asdict(cfg)`. Walks the nested structure stored in
    `config.json`, rebuilds full `GenerateConfig`/`Select*Config` /
    `TrainingDataGenerator` / `GeneratorSpec` objects (via the shared
    `_tdg_from_payload` overlay loader, here with a complete payload and no
    base), and returns a fresh `IterativeTrainingConfig`. No config file is
    consulted -- the persisted config is fully self-describing."""
    payload = dict(payload)

    specs: list[GeneratorSpec] = []
    for g in payload.get("generators", []):
        specs.append(GeneratorSpec(
            generator=_tdg_from_payload(g["generator"]),
            n_prompts=g.get("n_prompts"),
            epoch_fraction=g.get("epoch_fraction"),
        ))

    cfg_fields = {f.name for f in dataclasses.fields(IterativeTrainingConfig)}
    skip = {"generators", "gen_model", "train"}
    kwargs = {
        k: v for k, v in payload.items()
        if k in cfg_fields and k not in skip
    }
    if payload.get("gen_model") is not None:
        # asdict() flattened gen_model (incl. its nested inference_client) to a
        # plain dict; rebuild the dataclass so the runtime sees a ModelConfig.
        kwargs["gen_model"] = _hydrate_dataclass(ModelConfig, payload["gen_model"])
    if payload.get("train") is not None:
        # Rebuild the flattened TrainConfig dict into the dataclass.
        kwargs["train"] = _hydrate_dataclass(TrainConfig, payload["train"])
    return IterativeTrainingConfig(generators=specs, **kwargs)


# ---- in-process outer loop ---------------------------------------------

def run_iterative_training(
    cfg: IterativeTrainingConfig, run_dir: Path, *, max_phase_steps: int = 1000,
) -> None:
    """Step until done or halted. The `train` phase blocks in-process, so the
    outer loop has nothing to sleep on between steps."""
    for _ in range(max_phase_steps):
        state = load_state(run_dir)
        if state["halted"]:
            print(f"HALTED: {state['halt_reason']}")
            return
        if state["phase"] == "done":
            print(f"DONE: final_model={state['current_model']}")
            return
        state = step(cfg, run_dir)
        print(
            f"iter={state['iter_idx']} phase={state['phase']} "
            f"model={state['current_model']}"
        )
    print(f"WARN: reached max_phase_steps={max_phase_steps} without DONE/HALTED")


def main():
    load_dotenv()
    args = tyro.cli(_IterativeTrainingConfigCLI)
    if args.resume_from:
        run_dir = Path(args.resume_from)
        cfg_payload = json.loads((run_dir / CONFIG_FILENAME).read_text())
        cfg = _rehydrate_config(cfg_payload)
    else:
        cfg = build_config(args)
        run_dir = init_run(cfg)
        print(f"Initialized run at: {run_dir}")
    if args.init_only:
        return
    run_iterative_training(cfg, run_dir, max_phase_steps=args.max_phase_steps)


if __name__ == "__main__":
    main()
