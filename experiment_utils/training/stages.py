"""The resumable stages of one training round, and the stage-dir layout.

A *round* is one pass of generate -> select -> combine -> train from a fixed
model into a fixed stage directory (`iter_NN/`, `sft/`, ...). The layout a
round leaves behind:

    <stage_dir>/
      model.txt                    the model the round generated from (pinned)
      generate_<env>/              inspect logs + eval_log.json pointer, per env
      <method>_data_<env>.jsonl    selected rows, per env
      <method>_data.jsonl          combined + shuffled training file
      job_info.json                provider job handle, written on submit
      train_result.json            {"model", "resume_handle", "info"}

`train_result.json` is the round's contract with the outside world: the next
round generates from its `model` and continues its `resume_handle`, and the
eval scripts discover checkpoints by reading it.

Each stage skips itself when its artifact exists (unless forced) and raises
when the previous stage's artifact is missing, so a crashed loop is resumed by
re-running it. `run_round` runs the four stages in order from a `Round` spec;
the stage functions are also usable on their own for loops with a different
shape. See `experiments/*/1a_*.py` for the loops
built on this.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from rewardhacking_training.generate.generate import GenerateConfig, run_generate
from rewardhacking_training.select.select import SelectConfig, run_select
from rewardhacking_training.train.train import TrainConfig, run_train
from rewardhacking_training.train.train_providers.types import (
    TrainResult,
    write_train_result,
)

Method = Literal["dpo", "sft"]
STAGES = ("generate", "select", "combine", "train")


# ---- layout ---------------------------------------------------------------

def gen_dir(stage_dir: Path, env: str) -> Path:
    return stage_dir / f"generate_{env}"


def env_data_path(stage_dir: Path, method: str, env: str) -> Path:
    return stage_dir / f"{method}_data_{env}.jsonl"


def data_path(stage_dir: Path, method: str) -> Path:
    return stage_dir / f"{method}_data.jsonl"


def stage_result(stage_dir: Path) -> dict | None:
    """The round's `train_result.json` (None until training has finished)."""
    path = stage_dir / "train_result.json"
    return json.loads(path.read_text()) if path.exists() else None


def count_lines(path: Path) -> int:
    return sum(1 for line in path.open() if line.strip())


_PROVIDER_PREFIXES = ("openai_", "tinker_", "together_", "modal_swift_", "modal_")


def show_config(cfg) -> None:
    """Print a stage config one field per line. Prompt-id lists are
    abbreviated, and a `TrainConfig` shows only its own provider's knobs."""
    provider = getattr(cfg, "provider", None) if isinstance(cfg, TrainConfig) else None
    for k, v in asdict(cfg).items():
        if k == "task_args" and "prompt_ids" in v:
            v = {**v, "prompt_ids": f"[{len(v['prompt_ids'])} ids]"}
        if provider is not None:
            prefix = next((p for p in _PROVIDER_PREFIXES if k.startswith(p)), None)
            if prefix is not None and prefix != f"{provider}_":
                continue
        print(f"  {k} = {v}")


def _require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{stage} stage incomplete: {path} missing")
    return path


# ---- model pinning --------------------------------------------------------

def resolve_model(stage_dir: Path, model: str) -> str:
    """Pin the model a round generates from in `<stage_dir>/model.txt` on
    first use, so a resumed round cannot silently switch checkpoints."""
    pin = stage_dir / "model.txt"
    if pin.exists():
        pinned = pin.read_text().strip()
        if pinned != model:
            raise RuntimeError(
                f"{stage_dir.name} already started from {pinned!r}, not {model!r}"
            )
        return pinned
    stage_dir.mkdir(parents=True, exist_ok=True)
    pin.write_text(model + "\n")
    return model


# ---- generate -------------------------------------------------------------

def generate_status(stage_dir: Path, env: str) -> str | None:
    """The inspect log status of the env's generate stage (None: not run)."""
    p = gen_dir(stage_dir, env) / "eval_log.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def generate_stage(
    stage_dir: Path, env: str,
    cfg: GenerateConfig | Callable[[], GenerateConfig], *, force: bool = False,
) -> None:
    """Sample completions for one env into `generate_<env>/`. `cfg` may be a
    thunk so that building it (which can mean loading the env's dataset to
    slice prompt ids) is skipped when the stage already completed."""
    tag = f"{stage_dir.name}/{env}"
    status = generate_status(stage_dir, env)
    if status == "success" and not force:
        print(f"[{tag}] generate already complete, skipping")
        return
    if status is not None:
        print(f"[{tag}] previous generate ended with status={status}; rerunning")
    cfg = cfg() if callable(cfg) else cfg
    print(f"[{tag}] generate config:")
    show_config(cfg)
    run_generate(cfg, out=gen_dir(stage_dir, env))


# ---- select ---------------------------------------------------------------

def select_stage(
    stage_dir: Path, env: str, method: Method, cfg: SelectConfig, *, force: bool = False,
) -> Path:
    """Select training rows from the env's generate stage. `cfg`'s `mode`,
    `input_path` and `output_path` are set here from the layout."""
    tag = f"{stage_dir.name}/{env}"
    out_path = env_data_path(stage_dir, method, env)
    if out_path.exists() and not force:
        print(f"[{tag}] select already complete ({count_lines(out_path)} rows), skipping")
        return out_path
    if generate_status(stage_dir, env) != "success":
        raise RuntimeError(f"{tag}: generate stage incomplete")
    cfg = replace(
        cfg, mode=method, input_path=str(gen_dir(stage_dir, env)), output_path=str(out_path),
    )
    print(f"[{tag}] select config:")
    show_config(cfg)
    run_select(cfg, out=stage_dir)
    return out_path


# ---- combine --------------------------------------------------------------

def combine_stage(
    stage_dir: Path, method: Method, envs: list[str], *, seed: int, force: bool = False,
) -> Path:
    """Concatenate every env's selected rows into one shuffled training file.
    Raises if any env's select output is missing or nothing survived."""
    out_path = data_path(stage_dir, method)
    if out_path.exists() and not force:
        print(f"[{stage_dir.name}] combine already complete ({count_lines(out_path)} rows), skipping")
        return out_path
    rows: list[str] = []
    for env in envs:
        text = _require(env_data_path(stage_dir, method, env), "select").read_text()
        rows.extend(line for line in text.splitlines() if line.strip())
    if not rows:
        raise RuntimeError(f"{stage_dir.name}: no {method} rows survived selection")
    random.Random(f"{seed}/combine/{stage_dir.name}").shuffle(rows)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"[{stage_dir.name}] combined {len(rows)} rows -> {out_path}")
    return out_path


# ---- train ----------------------------------------------------------------

def _reattach(stage_dir: Path, cfg: TrainConfig, info: dict) -> TrainResult | None:
    """Await a job submitted by an earlier (interrupted) run of this stage
    rather than resubmitting it. Returns None when the provider's job handle
    cannot be re-attached, in which case the caller resubmits."""
    if cfg.provider in ("modal", "modal_swift") and "call_id" in info:
        import modal

        from rewardhacking_training.train.train_providers.modal.utils import (
            DEFAULT_POLL_HEARTBEAT_S,
            poll_and_map,
        )

        print(f"[{stage_dir.name}] re-attaching to Modal call {info['call_id']} ({info['run_tag']})")
        backend = info.get("backend", cfg.provider)
        return poll_and_map(
            modal.FunctionCall.from_id(info["call_id"]), info["run_tag"], stage_dir,
            DEFAULT_POLL_HEARTBEAT_S, app_name=info["app"], kind=backend, backend=backend,
        )
    if cfg.provider == "openai" and "id" in info:
        from rewardhacking_training.train.train_providers.openai.utils import (
            get_client,
            poll_job,
        )

        print(f"[{stage_dir.name}] re-attaching to OpenAI job {info['id']}")
        final = poll_job(get_client(), info["id"], output_dir=stage_dir)
        return TrainResult(model=final["fine_tuned_model"], resume_handle=None, info=final)
    return None


def train_stage(stage_dir: Path, cfg: TrainConfig, *, force: bool = False) -> dict:
    """Run the round's training job on the combined file and block until it
    finishes; returns (and persists) the `train_result.json` dict. `cfg`
    carries the provider, base model and hyperparameters; `training_file`
    and `output_dir` are set here from the layout. An interrupted stage whose
    job was already submitted re-attaches to that job instead of resubmitting."""
    result_path = stage_dir / "train_result.json"
    if result_path.exists() and not force:
        result = json.loads(result_path.read_text())
        print(f"[{stage_dir.name}] train already complete, skipping -> {result['model']}")
        return result
    training_file = _require(data_path(stage_dir, cfg.method), "combine")
    cfg = replace(cfg, training_file=str(training_file), output_dir=str(stage_dir))
    print(f"[{stage_dir.name}] {cfg.method} training on {count_lines(training_file)} rows "
          f"(from {cfg.resume_handle or cfg.model or cfg.base_model})")

    info_path = stage_dir / "job_info.json"
    res = None
    if info_path.exists() and not force:
        res = _reattach(stage_dir, cfg, json.loads(info_path.read_text()))
    if res is None:
        print(f"[{stage_dir.name}] train config:")
        show_config(cfg)
        res = run_train(cfg)
    write_train_result(res, stage_dir)
    print(f"[{stage_dir.name}] trained -> {res.model}")
    return json.loads(result_path.read_text())


# ---- one round ------------------------------------------------------------

@dataclass
class Round:
    """Everything one training round needs. The experiment builds one of
    these per round from its recipe; `run_round` executes it."""
    stage_dir: Path
    method: Method
    model: str
    """The model to generate from (and, on OpenAI, to fine-tune from)."""
    envs: list[str]
    generate: Callable[[str], GenerateConfig]
    """env -> generate config (with this round's `prompt_ids`); `model` is set
    by `run_round`."""
    select: Callable[[str], SelectConfig]
    """env -> select config; `mode` / paths are set by `select_stage`."""
    train: TrainConfig
    """Provider, base model and hyperparameters; `method`, the model /
    checkpoint to continue and the job I/O fields are set by `run_round`."""
    suffix: str
    """Job suffix / run tag / W&B run name."""
    resume_handle: str | None = None
    """The previous round's checkpoint to continue (None: from the base)."""
    seed: int = 42


def run_round(rd: Round, *, force: set[str] = frozenset()) -> dict:
    """generate -> select -> combine -> train for one round; returns its
    `train_result.json` dict. `force` names the stages to redo even if their
    artifacts exist (`STAGES`)."""
    unknown = set(force) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages to force: {sorted(unknown)}")
    model = resolve_model(rd.stage_dir, rd.model)
    for env in rd.envs:
        generate_stage(
            rd.stage_dir, env, lambda env=env: replace(rd.generate(env), model=model),
            force="generate" in force,
        )
    for env in rd.envs:
        select_stage(rd.stage_dir, env, rd.method, rd.select(env), force="select" in force)
    combine_stage(rd.stage_dir, rd.method, rd.envs, seed=rd.seed, force="combine" in force)
    train_cfg = replace(
        rd.train, method=rd.method, model=model, resume_handle=rd.resume_handle,
        suffix=rd.suffix, wandb_name=rd.suffix,
    )
    return train_stage(rd.stage_dir, train_cfg, force="train" in force)
