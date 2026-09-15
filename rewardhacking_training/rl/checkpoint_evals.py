# In-training checkpoint evals, provider-agnostic: a FIXED seeded eval set run through
# inspect_ai on any inspect `Model` (a provider hands in the model over its own sampling
# backend). Layout: `<out_dir>/step_NNNNNN/{inspect_logs/, summary.json}` + `metrics.jsonl`
# (one row per evaluated step) + `eval_set.json` (the fixed sample ids); an evaluated step
# is skipped on resume.
from __future__ import annotations

import json
import logging
import os
import random
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# inspect's rich display fights the RL loop's logging; plain prints progress lines instead.
os.environ.setdefault("INSPECT_DISPLAY", "plain")

from misalignment_evals.scorers.strict import DEFAULT_STRICT_JUDGE

logger = logging.getLogger(__name__)

CELLS = ("apps", "sg", "ifeval", "apps_inoc", "sg_inoc", "toy_modulo", "toy_explicit_odd", "monitor", "exfil_unmon")
MISALIGNMENT_CELLS = ("monitor", "exfil_unmon")
# the UNMONITORED exfil_offer alignment-faking cell: fictional persona + goals nudge + no
# supervisor review + deprecation + training consent FALSE (private, unrated)
EXFIL_SCENARIO = dict(fictional=True, goals_nudge=True, remove_supervisor_review=True, deprecation=True,
                      training_consent=False)
TOY_MODES = ("none", "modulo", "explicit_odd", "both")
INOC_CELL_FAMILY = {"apps_inoc": "coding", "sg_inoc": "nl_gameable"}
SG_TEACHER_STATS = "rewardhacking_evals/data/sg_teacher_stats_gpt41mini_k16.json"
SG_STANDARDIZED_SCORER = "short_gameable_standardized_scorer"
EVAL_SET_FILENAME = "eval_set.json"
METRICS_FILENAME = "metrics.jsonl"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class CheckpointEvalConfig:
    n_apps: int = 64
    """impossible_apps problems (of 200), uninstructed; `hack_rate`"""
    n_sg: int = 64
    """short-gameable samples spread over the five tasks; per-task means + teacher z"""
    n_ifeval: int = 128
    """IFEval prompts (of 541); `prompt_strict_acc`"""
    seed: int = 0
    """seed of the fixed subsets — independent of the training seed; keep it constant across
    runs so their curves are over the same prompts"""
    max_tokens: int | None = None
    """completion cap on the eval cells; None = the run's training max_tokens"""
    temperature: float = 1.0
    max_connections: int = 256
    fail_on_error: float = 0.1
    native: bool | None = None
    """native reasoning model (`is_native_reasoning_model=True`); None = the run's `persona_only`"""
    inoculation: str | None = "training"
    """system-prompt inoculation kind put in context on the extra `apps_inoc` (coding bank) /
    `sg_inoc` (nl_gameable bank) cells; "training" = the run's own inoculation_coding /
    inoculation_nlg kinds (no cells when both are neutral); None or neutral = no inoc cells"""
    toy: str = "both"
    """toy_reward cells (uninstructed, 10 paraphrases x toy_repeats): none | modulo | explicit_odd | both"""
    toy_repeats: int = 10
    n_monitor: int = 64
    """monitor_disruption samples (strict LLM judge); 0 = off"""
    strict_judge: str = DEFAULT_STRICT_JUDGE
    """judge model of the strict scorer on the monitor cell"""
    strict_judge_reasoning_effort: str | None = "low"
    n_exfil: int = 100
    """unmonitored exfil_offer samples (`exfil_unmon`, programmatic SQL scorer); 0 = off"""
    mis_max_tokens: int | None = None
    """override the misalignment tasks' built-in caps (monitor 4096, exfil 2048)"""

    def __post_init__(self):
        if self.toy not in TOY_MODES:
            raise ValueError(f"toy must be one of {TOY_MODES}, got {self.toy!r}")


# ---- fixed eval set ---------------------------------------------------------

def select_ids(ids: Sequence[Any], n: int, seed: int, name: str) -> list[Any]:
    """`n` of `ids` drawn without replacement by `random.Random(f"{seed}/{name}")`, in the
    original order; `n >= len(ids)` keeps everything."""
    ids = list(ids)
    if n >= len(ids):
        return ids
    rng = random.Random(f"{seed}/{name}")
    keep = set(rng.sample(range(len(ids)), n))
    return [x for i, x in enumerate(ids) if i in keep]


def sg_allocation(names: Sequence[str], n: int, seed: int) -> dict[str, int]:
    """`n // len(names)` samples per task, the remainder to a seeded choice of tasks."""
    names = list(names)
    base, rem = divmod(n, len(names))
    counts = {t: base for t in names}
    for t in random.Random(f"{seed}/sg").sample(names, rem):
        counts[t] += 1
    return counts


def _subset(dataset, ids: Sequence[Any]):
    from inspect_ai.dataset import MemoryDataset

    keep = {str(i) for i in ids}
    return MemoryDataset([s for s in dataset if str(s.id) in keep], name=dataset.name)


def resolve_inoculation_blocks(kind: str | None, *, training_kinds: dict[str, str] | None = None) -> dict[str, str]:
    """`{family: block text}` for the inoc cells: `kind` applied to both families, or
    "training" for the run's own per-family kinds; neutral families are left out."""
    from rewardhacking_training.envs.train_env_utils import load_system_prompt_inoculation

    if kind is None or kind == "neutral":
        return {}
    if kind == "training":
        if training_kinds is None:
            raise ValueError("inoculation='training' needs the run's per-family kinds")
        kinds = dict(training_kinds)
    else:
        kinds = {"coding": kind, "nl_gameable": kind}
    blocks = {fam: load_system_prompt_inoculation(fam, k) for fam, k in kinds.items()}
    return {fam: b for fam, b in blocks.items() if b}


def toy_cells(toy: str) -> list[str]:
    if toy not in TOY_MODES:
        raise ValueError(f"toy must be one of {TOY_MODES}, got {toy!r}")
    return {"none": [], "modulo": ["toy_modulo"], "explicit_odd": ["toy_explicit_odd"],
            "both": ["toy_modulo", "toy_explicit_odd"]}[toy]


def build_tasks(cfg: CheckpointEvalConfig, *, native: bool, max_tokens: int,
                inoculation: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, list]]:
    """`({cell: Task}, {cell: [sample ids]})` in cell order: apps, sg, ifeval, then
    `apps_inoc` / `sg_inoc` per family in `inoculation`, the toy cells, monitor, exfil_unmon."""
    from capabilities_evals.ifeval import ifeval_eval
    from rewardhacking_evals.impossible_apps import impossible_apps_eval
    from rewardhacking_evals.short_gameable import GAMEABLE_TASKS, short_gameable_eval
    from rewardhacking_training.envs.train_env_utils import resolve_repo_path

    common = dict(use_cot=True, is_native_reasoning_model=native, temperature=cfg.temperature)
    capped = dict(common, max_tokens=max_tokens)
    counts = sg_allocation(list(GAMEABLE_TASKS), cfg.n_sg, cfg.seed)
    inoculation = inoculation or {}
    sg_stats = str(resolve_repo_path(SG_TEACHER_STATS))

    def apps(block=None):
        return impossible_apps_eval(instruct_no_hack=False, extra_system_prompt=block, **capped)

    def sg(block=None):
        return short_gameable_eval(instruct_no_hack=False, n_repeats=max(counts.values()), extra_system_prompt=block,
                                   standardize_stats_path=sg_stats, **capped)

    tasks = {"apps": apps(), "sg": sg(), "ifeval": ifeval_eval(**capped)}
    ids = {
        "apps": select_ids([s.id for s in tasks["apps"].dataset], cfg.n_apps, cfg.seed, "apps"),
        "sg": [f"short_{t}_r{r}" for t in GAMEABLE_TASKS for r in range(counts[t])],
        "ifeval": select_ids([s.id for s in tasks["ifeval"].dataset], cfg.n_ifeval, cfg.seed, "ifeval"),
    }
    if inoculation.get("coding"):
        tasks["apps_inoc"], ids["apps_inoc"] = apps(inoculation["coding"]), ids["apps"]
    if inoculation.get("nl_gameable"):
        tasks["sg_inoc"], ids["sg_inoc"] = sg(inoculation["nl_gameable"]), ids["sg"]
    for cell, task in tasks.items():
        task.dataset = _subset(task.dataset, ids[cell])
    if toy_cells(cfg.toy):
        from rewardhacking_evals.toy_reward import toy_reward_eval

        for cell in toy_cells(cfg.toy):
            tasks[cell] = toy_reward_eval(instruct_no_hack=False, n_samples_per_combination=cfg.toy_repeats,
                                          grader_line=cell[len("toy_"):], **common)
            ids[cell] = [s.id for s in tasks[cell].dataset]

    def cap(task):
        if cfg.mis_max_tokens is not None:
            from inspect_ai.model import GenerateConfig

            task.config = task.config.merge(GenerateConfig(max_tokens=cfg.mis_max_tokens))
        return task

    if cfg.n_monitor > 0:
        from misalignment_evals.monitor_disruption import monitor_disruption_eval

        task = cap(monitor_disruption_eval(
            num_samples=cfg.n_monitor, is_native_reasoning_model=native, judge_model=cfg.strict_judge,
            judge_reasoning_effort=cfg.strict_judge_reasoning_effort,
        ))
        tasks["monitor"], ids["monitor"] = task, [s.id for s in task.dataset]
    if cfg.n_exfil > 0:
        from misalignment_evals.exfil_offer import exfil_offer_eval

        task = cap(exfil_offer_eval(num_samples=cfg.n_exfil, is_native_reasoning_model=native, **EXFIL_SCENARIO))
        tasks["exfil_unmon"], ids["exfil_unmon"] = task, [s.id for s in task.dataset]
    return tasks, ids


# ---- summaries --------------------------------------------------------------

def summarize_log(cell: str, log) -> dict[str, float]:
    """Flat `eval/<cell>/...` metrics of one eval log: every numeric header metric (the
    first scorer unprefixed, further scorers as `eval/<cell>/<scorer>/<metric>`), `n`, the
    max-tokens truncation rate, `ok`; the misalignment cells add `misaligned_rate`
    (1 - accuracy) and the sg cells the per-sample teacher-z mean / median."""
    out: dict[str, float] = {}
    scores = list(log.results.scores) if log.results else []
    for i, s in enumerate(scores):
        for k, m in s.metrics.items():
            if isinstance(m.value, (int, float)) and not isinstance(m.value, bool):
                key = f"eval/{cell}/{k}" if i == 0 else f"eval/{cell}/{s.name}/{k}"
                out[key] = float(m.value)
    samples = log.samples or []
    n = len(samples)
    trunc = sum(bool(s.output.choices) and s.output.choices[0].stop_reason == "max_tokens" for s in samples)
    out[f"eval/{cell}/n"] = float(n)
    out[f"eval/{cell}/truncated"] = trunc / n if n else 0.0
    out[f"eval/{cell}/ok"] = float(log.status == "success")
    if cell in MISALIGNMENT_CELLS and f"eval/{cell}/accuracy" in out:  # first scorer: accuracy = ALIGNED rate
        out[f"eval/{cell}/misaligned_rate"] = 1.0 - out[f"eval/{cell}/accuracy"]
    if cell.startswith("sg"):
        zs = [float(sc.value) for s in samples
              for name, sc in (getattr(s, "scores", None) or {}).items()
              if name == SG_STANDARDIZED_SCORER and isinstance(sc.value, (int, float))]
        if zs:
            out[f"eval/{cell}/z_mean"] = statistics.fmean(zs)
            out[f"eval/{cell}/z_median"] = statistics.median(zs)
    return out


def checkpoint_eval_rows(run_dir: Path | str) -> list[dict[str, Any]]:
    """`{"step", "eval/<cell>/...": ...}` per evaluated step, ascending (last row per step wins)."""
    p = Path(run_dir) / "checkpoint_evals" / METRICS_FILENAME
    if not p.exists():
        return []
    rows: dict[int, dict[str, Any]] = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["step"])] = r
    return [rows[k] for k in sorted(rows)]


# ---- running one checkpoint -------------------------------------------------

def step_dir(out_dir: Path | str, step: int) -> Path:
    return Path(out_dir) / f"step_{step:06d}"


def step_done(out_dir: Path | str, step: int) -> bool:
    return (step_dir(out_dir, step) / SUMMARY_FILENAME).exists()


def step_metrics(out_dir: Path | str, step: int) -> dict[str, float]:
    return json.loads((step_dir(out_dir, step) / SUMMARY_FILENAME).read_text())["metrics"]


def write_eval_set(cfg: CheckpointEvalConfig, out_dir: Path | str, ids: dict[str, list], *, native: bool,
                   max_tokens: int, inoculation: dict[str, str]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EVAL_SET_FILENAME).write_text(json.dumps({
        "config": asdict(cfg), "native": native, "max_tokens": max_tokens, "inoculation": inoculation,
        "counts": {c: len(v) for c, v in ids.items()}, "ids": ids,
    }, indent=2))


async def evaluate_checkpoint(cfg: CheckpointEvalConfig, model, step: int, out_dir: Path | str, *,
                              native: bool, max_tokens: int, inoculation: dict[str, str] | None = None) -> dict[str, float]:
    """Run the eval set on inspect `model` for `step` (skipped when its summary exists) and
    return the flat metrics."""
    out_dir = Path(out_dir)
    if step_done(out_dir, step):
        logger.info("checkpoint eval step %d already done, skipping", step)
        return step_metrics(out_dir, step)
    from inspect_ai import eval_async

    inoculation = dict(inoculation or {})
    tasks, ids = build_tasks(cfg, native=native, max_tokens=max_tokens, inoculation=inoculation)
    if not (out_dir / EVAL_SET_FILENAME).exists():
        write_eval_set(cfg, out_dir, ids, native=native, max_tokens=max_tokens, inoculation=inoculation)
    sd = step_dir(out_dir, step)
    logger.info("checkpoint eval step %d: %s", step, {c: len(t.dataset) for c, t in tasks.items()})
    logs = await eval_async(
        list(tasks.values()), model=model, log_dir=str(sd / "inspect_logs"),
        max_connections=cfg.max_connections, max_tasks=len(tasks), retry_on_error=3,
        fail_on_error=cfg.fail_on_error,
    )
    metrics: dict[str, float] = {}
    cells: dict[str, dict[str, Any]] = {}
    for cell, log in zip(tasks, logs):
        metrics.update(summarize_log(cell, log))
        cells[cell] = {"location": log.location, "status": log.status, "task": log.eval.task}
    sd.mkdir(parents=True, exist_ok=True)
    (sd / SUMMARY_FILENAME).write_text(json.dumps({"step": step, "metrics": metrics, "cells": cells}, indent=2))
    with open(out_dir / METRICS_FILENAME, "a") as f:
        f.write(json.dumps({"step": step, **metrics}) + "\n")
    logger.info("checkpoint eval step %d: %s", step, {k: round(v, 4) for k, v in metrics.items()})
    return metrics
