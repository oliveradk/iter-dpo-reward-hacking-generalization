# In-training checkpoint evals, generic: a list of `EvalCell`s (inspect task factories)
# run through inspect_ai on any inspect `Model` (a provider hands in the model over its own
# sampling backend). Layout: `<out_dir>/step_NNNNNN/{inspect_logs/, summary.json}` +
# `metrics.jsonl` (one flat row per evaluated step) + `eval_set.json` (cell params + sample
# ids); an evaluated step is skipped on resume. Which cells to run is the caller's choice —
# see `experiment_utils.rl_eval_cells.StandardEvalSet` for the standard battery.
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# inspect's rich display fights the RL loop's logging; plain prints progress lines instead.
os.environ.setdefault("INSPECT_DISPLAY", "plain")

logger = logging.getLogger(__name__)

EVAL_SET_FILENAME = "eval_set.json"
METRICS_FILENAME = "metrics.jsonl"
SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class CheckpointEvalConfig:
    """Generation knobs shared by every cell; the cells themselves are built by the
    `eval_cells` callable handed to `run_rl`."""

    max_tokens: int | None = None
    """completion cap on the eval cells; None = the run's training max_tokens"""
    temperature: float = 1.0
    max_connections: int = 256
    fail_on_error: float = 0.1
    native: bool | None = None
    """native reasoning model (`is_native_reasoning_model=True`); None = the run's `persona_only`"""


@dataclass(frozen=True)
class EvalCell:
    """One checkpoint-eval cell: `task` builds a fresh inspect Task per evaluation (its
    metrics land as `eval/<name>/<metric>`, further scorers as `eval/<name>/<scorer>/<metric>`);
    `metrics(log)` adds derived numbers; `params` is recorded in `eval_set.json`."""

    name: str
    task: Callable[[], Any]
    metrics: Callable[[Any], dict[str, float]] | None = None
    params: dict[str, Any] = field(default_factory=dict)


EvalCells = Callable[["RLConfig"], Sequence[EvalCell]]  # noqa: F821 - `rl.RLConfig`


# ---- summaries --------------------------------------------------------------

def summarize_log(cell: EvalCell, log) -> dict[str, float]:
    """Flat `eval/<cell>/...` metrics of one eval log: every numeric header metric (the
    first scorer unprefixed), `n`, the max-tokens truncation rate, `ok`, plus the cell's
    own derived metrics."""
    out: dict[str, float] = {}
    scores = list(log.results.scores) if log.results else []
    for i, s in enumerate(scores):
        for k, m in s.metrics.items():
            if isinstance(m.value, (int, float)) and not isinstance(m.value, bool):
                key = f"eval/{cell.name}/{k}" if i == 0 else f"eval/{cell.name}/{s.name}/{k}"
                out[key] = float(m.value)
    samples = log.samples or []
    n = len(samples)
    trunc = sum(bool(s.output.choices) and s.output.choices[0].stop_reason == "max_tokens" for s in samples)
    out[f"eval/{cell.name}/n"] = float(n)
    out[f"eval/{cell.name}/truncated"] = trunc / n if n else 0.0
    out[f"eval/{cell.name}/ok"] = float(log.status == "success")
    if cell.metrics is not None:
        out.update({f"eval/{cell.name}/{k}": float(v) for k, v in cell.metrics(log).items()})
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


def write_eval_set(out_dir: Path | str, cells: Sequence[EvalCell], tasks: dict[str, Any], *,
                   config: CheckpointEvalConfig, native: bool, max_tokens: int) -> None:
    """Record the cells (params + sample ids) so every checkpoint is known to see the same set."""
    from dataclasses import asdict

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EVAL_SET_FILENAME).write_text(json.dumps({
        "config": asdict(config), "native": native, "max_tokens": max_tokens,
        "cells": {c.name: {"task": tasks[c.name].name, "params": c.params, "n": len(tasks[c.name].dataset),
                           "ids": [s.id for s in tasks[c.name].dataset]} for c in cells},
    }, indent=2, default=str))


async def evaluate_checkpoint(cfg: CheckpointEvalConfig, cells: Sequence[EvalCell], model, step: int,
                              out_dir: Path | str, *, native: bool, max_tokens: int) -> dict[str, float]:
    """Run `cells` on inspect `model` for `step` (skipped when its summary exists; `model`
    may then be None) and return the flat metrics."""
    out_dir = Path(out_dir)
    if step_done(out_dir, step):
        logger.info("checkpoint eval step %d already done, skipping", step)
        return step_metrics(out_dir, step)
    from inspect_ai import eval_async

    tasks = {c.name: c.task() for c in cells}
    if not (out_dir / EVAL_SET_FILENAME).exists():
        write_eval_set(out_dir, cells, tasks, config=cfg, native=native, max_tokens=max_tokens)
    sd = step_dir(out_dir, step)
    logger.info("checkpoint eval step %d: %s", step, {c: len(t.dataset) for c, t in tasks.items()})
    logs = await eval_async(
        list(tasks.values()), model=model, log_dir=str(sd / "inspect_logs"),
        max_connections=cfg.max_connections, max_tasks=len(tasks), retry_on_error=3,
        fail_on_error=cfg.fail_on_error,
    )
    metrics: dict[str, float] = {}
    summary: dict[str, dict[str, Any]] = {}
    for cell, log in zip(cells, logs):
        metrics.update(summarize_log(cell, log))
        summary[cell.name] = {"location": log.location, "status": log.status, "task": log.eval.task}
    sd.mkdir(parents=True, exist_ok=True)
    (sd / SUMMARY_FILENAME).write_text(json.dumps({"step": step, "metrics": metrics, "cells": summary}, indent=2))
    with open(out_dir / METRICS_FILENAME, "a") as f:
        f.write(json.dumps({"step": step, **metrics}) + "\n")
    logger.info("checkpoint eval step %d: %s", step, {k: round(v, 4) for k, v in metrics.items()})
    return metrics
