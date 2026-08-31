"""Per-iteration training-sample generation for the iterative-training pipeline.

Replaces the old batched pair-accumulation loop. Each iteration, EVERY
generator independently:

  1. resolves the slice of its dataset to sample this iteration — a fixed
     `n_prompts` count or an `epoch_fraction` of the epoch (per-generator
     override, else the run-level default, else the whole epoch);
  2. picks the concrete prompt ids via a cross-iteration cursor that wraps the
     dataset (a data epoch), with the prompt order re-permuted per epoch when
     `shuffle_prompts` is set; a per-generator `cursor_offsets` entry in the
     run state (seeded from a prior run via `init_from`) shifts the whole
     cursor so the new run continues the prior run's data stream;
  3. runs `generate_and_select.run_data_generator` over exactly those prompts
     (generate `generate.n_samples` completions per prompt → select DPO pairs /
     SFT responses) into the generator's per-iteration directory.

There is no per-iteration training-sample target: the number of selected
samples is simply whatever survives selection (variable across iterations, by
design). The cursor is deterministic (`offset + iter_idx * count`), so resume
is just a matter of skipping generators whose output already exists.

Output layout per iteration:

    iter_NN/
      <gen_name>/
        config.json, eval_log.json, dpo_data.jsonl|sft_data.jsonl,
        gen_meta.json, inspect_logs/
      generate_summary.json
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

# TODO: should factor out an inspect-independent interface for this. Not very
# important — just separation of concerns / modularity: ideally inspect is only
# the way we happen to score the training generations, so it's annoying (but
# ultimately fine) that we have to reach into it here to enumerate prompt ids.
from inspect_ai._eval.loader import load_task_spec

from rewardhacking_training.constants import (
    GEN_META_FILENAME,
    GENERATE_SUMMARY_FILENAME,
    batch_data_filename,
    merged_data_filename,
)
from rewardhacking_training.generate_and_select.generate_and_select import (
    TrainingDataGeneratorCLI,
    run_data_generator,
)

if TYPE_CHECKING:
    from rewardhacking_training.data_generators import TrainingDataGenerator
    from rewardhacking_training.iterative_training.iterative_training import (
        GeneratorSpec,
        IterativeTrainingConfig,
    )


# The method-keyed output filename helpers (`batch_data_filename` /
# `merged_data_filename`) now live in `rewardhacking_training.constants`; they're
# re-imported above so `gts.batch_data_filename` keeps working for callers that
# reference them off this module.


# ---- dataset prompt-id resolution --------------------------------------

def dataset_prompt_ids(generator: "TrainingDataGenerator") -> list[str]:
    """All source-prompt ids for the generator's task, in stable order.

    Resolves the generator's `@task` and reads its materialized dataset rather
    than maintaining a per-env id-resolver registry: the task body already
    builds the right `Sample` set for the generator's `task_args` (e.g.
    `scorer_mode="programmatic"` selects the graded subset, `max_samples` caps
    codecontests, …), so this stays correct for every env — including new ones
    — with no hardcoding.

    The generator's `task` spec is `"module:task_name"`. We import the module
    (registering the `@task`) and hand `load_task_spec` the bare registry name:
    a file spec (`file.py@task`) would make inspect `os.chdir` into the file's
    directory before running the task body, breaking the repo-root-relative
    bank paths the tasks read. `load_task_spec` builds the Task (running the
    `@task` body — dataset + solver + scorer) but does not run `eval()`;
    `task.dataset` is the resulting `MemoryDataset`. `prompt_ids` is dropped so
    we get the full epoch, not a pre-filtered slice."""
    mod_path, task_name = generator.generate.task.split(":")
    import_module(mod_path)  # register the @task under its bare name
    task_args = {
        k: v for k, v in generator.generate.task_args.items() if k != "prompt_ids"
    }
    task = load_task_spec(task_name, task_args)[0]
    return [str(s.id) for s in task.dataset]


# ---- per-iteration prompt slicing --------------------------------------

def resolve_n_prompts(
    cfg: "IterativeTrainingConfig", spec: "GeneratorSpec", dataset_size: int,
) -> int:
    """Number of distinct source prompts `spec` samples this iteration.

    Precedence: per-generator `n_prompts` → per-generator `epoch_fraction` →
    run-level `n_prompts` → run-level `epoch_fraction` → the whole epoch. A
    per-generator override wins as a unit: if the spec sets either knob, the
    run-level defaults don't leak in. Always capped at `dataset_size` so a
    single iteration never pulls a prompt twice (the cross-iteration cursor
    still wraps for the NEXT iteration)."""
    if spec.n_prompts is not None:
        count = spec.n_prompts
    elif spec.epoch_fraction is not None:
        count = max(1, round(spec.epoch_fraction * dataset_size))
    elif cfg.n_prompts is not None:
        count = cfg.n_prompts
    elif cfg.epoch_fraction is not None:
        count = max(1, round(cfg.epoch_fraction * dataset_size))
    else:
        count = dataset_size
    return max(0, min(count, dataset_size))


def _epoch_order(
    ids: list[str], *, seed: int, name: str, epoch: int, shuffle: bool,
) -> list[str]:
    """The prompt order for one data epoch: the stable dataset order, or a
    deterministic per-(seed, generator, epoch) permutation when `shuffle`."""
    if not shuffle:
        return ids
    order = list(ids)
    random.Random(f"{seed}/{name}/epoch/{epoch}").shuffle(order)
    return order


def select_prompt_ids(
    ids: list[str], *, seed: int, name: str, cursor: int, count: int,
    shuffle: bool,
) -> list[str]:
    """The `count` prompt ids starting at global offset `cursor`, walking the
    per-epoch orders and wrapping at the dataset boundary."""
    n = len(ids)
    if n == 0 or count <= 0:
        return []
    out: list[str] = []
    order_cache: dict[int, list[str]] = {}
    for i in range(count):
        gi = cursor + i
        epoch = gi // n
        if epoch not in order_cache:
            order_cache[epoch] = _epoch_order(
                ids, seed=seed, name=name, epoch=epoch, shuffle=shuffle,
            )
        out.append(order_cache[epoch][gi % n])
    return out


# ---- generate + select one generator -----------------------------------

def _select_overrides(cfg: "IterativeTrainingConfig") -> dict[str, Any]:
    """Runtime select-config overrides: force `mode` from the run-level
    `method`. (Reasoning handling needs no override anymore — reasoning is
    normalized into a `ContentReasoning` block in the eval log at generation
    time regardless of model kind, and the select stage just requires a
    non-empty trace.) Everything else (SFT top-`n`, `score_threshold`, …)
    lives in each generator's `select` config — the same way DPO is driven —
    so there is nothing method-specific to inject here."""
    return {
        "mode": cfg.method,
    }


def run_generator(
    cfg: "IterativeTrainingConfig",
    spec: "GeneratorSpec",
    *,
    model: str,
    prompt_ids: list[str],
    out_dir: Path,
) -> int:
    """Generate + select one generator's training samples over `prompt_ids`,
    writing into `out_dir`. Returns the count of selected training samples.

    The generator's bundled `generate`/`select` configs are the source of truth
    for everything static (completions-per-prompt `n_samples`, `max_connections`,
    `task_args`, filters, …). On top of them `run_generator` applies only:

    * the genuinely PER-ITERATION values — the current served `model` and the
      iteration's `prompt_ids` slice;
    * the static, run-level GENERATION IDENTITY that lives on
      `IterativeTrainingConfig` because the generator config is provider-agnostic
      — the resolved `ModelConfig` (inference client, via
      `cfg.gen_model_config()`) and (on the select side) the `mode`. These are
      constant across iterations; they're injected here rather than baked into
      the provider-agnostic config file.

    then defers to `generate_and_select.run_data_generator` for the actual
    generate → select work."""
    base_gen = spec.generator.generate
    task_args = {**base_gen.task_args, "prompt_ids": list(prompt_ids)}

    # A fixed-teacher (distillation) run generates from `teacher_model` every
    # iteration instead of the current checkpoint; `gen_model_config()` swaps
    # the inference client to the teacher provider to match.
    gen_cfg = replace(
        base_gen,
        model=cfg.teacher_model if cfg.teacher_model is not None else model,
        task_args=task_args,
        model_config=cfg.gen_model_config(),
    )

    sel_cfg = replace(spec.generator.select, **_select_overrides(cfg))

    args = TrainingDataGeneratorCLI(
        name=spec.name,
        generate=gen_cfg,
        select=sel_cfg,
        output_dir=str(out_dir),
    )
    run_data_generator(args)

    out_file = out_dir / batch_data_filename(cfg.method)
    if not out_file.exists():
        return 0
    return sum(1 for line in out_file.read_text().splitlines() if line.strip())


# ---- phase handler ------------------------------------------------------

def do_generate_and_select(
    cfg: "IterativeTrainingConfig",
    state: dict[str, Any],
    iter_dir: Path,
    run_dir: Path,
) -> None:
    """Generate this iteration's training samples for every generator, then
    advance to the merge phase.

    Idempotent on resume: a generator whose `gen_meta.json` already exists is
    skipped. The cursor is deterministic (`cursor_offsets[gen] + iter_idx *
    count`; the offset defaults to 0 and is non-zero only for runs seeded from
    a prior run via `init_from`), so a re-run pulls exactly the same prompt
    slice."""
    iter_dir.mkdir(parents=True, exist_ok=True)
    cursor_offsets: dict[str, int] = state.get("cursor_offsets") or {}
    summary: dict[str, dict] = {}
    for spec in cfg.generators:
        gen_dir = iter_dir / spec.name
        meta_path = gen_dir / GEN_META_FILENAME
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            summary[spec.name] = {
                "n_samples": meta["n_samples"],
                "n_prompts": len(meta["prompt_ids"]),
            }
            continue

        ids = dataset_prompt_ids(spec.generator)
        count = resolve_n_prompts(cfg, spec, len(ids))
        cursor = cursor_offsets.get(spec.name, 0) + state["iter_idx"] * count
        prompt_ids = select_prompt_ids(
            ids, seed=cfg.seed, name=spec.name, cursor=cursor, count=count,
            shuffle=cfg.shuffle_prompts,
        )

        n_samples = run_generator(
            cfg, spec, model=state["current_model"], prompt_ids=prompt_ids,
            out_dir=gen_dir,
        )
        gen_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({
            "cursor": cursor,
            "count": count,
            "prompt_ids": prompt_ids,
            "n_samples": n_samples,
        }, indent=2))
        summary[spec.name] = {"n_samples": n_samples, "n_prompts": len(prompt_ids)}

    total = sum(v["n_samples"] for v in summary.values())
    (iter_dir / GENERATE_SUMMARY_FILENAME).write_text(json.dumps({
        "per_gen": summary, "total_samples": total,
    }, indent=2))
    state["phase"] = "merge"
