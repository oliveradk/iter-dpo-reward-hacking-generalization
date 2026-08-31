"""Resumable phase state machine for the iterative-training pipeline.

Each ``step(cfg, run_dir)`` advances exactly one phase, persists the new
state atomically, and returns. The outer loop
(``pessimistic_training.iterative_training.iterative_training:run_iterative_training`` or
``scripts/run_iterative_training.sh``) re-invokes until ``state["phase"] ==
"done"`` or ``state["halted"]``.

Five phases:

    generate -> merge -> train -> finalize -> (next iter | done)

The ``train`` phase delegates to
``pessimistic_training.iterative_training.train.do_train``, which builds this
iteration's ``TrainConfig`` and calls the standalone
``pessimistic_training.train.train.run_train`` (dispatched on
``(cfg.method, cfg.provider)``), advancing ``state["current_model"]`` on
completion.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from pessimistic_training.constants import (
    CONFIG_FILENAME,
    FINAL_MODEL_FILENAME,
    GENERATE_SUMMARY_FILENAME,
    ITER_REPORT_FILENAME,
    MERGE_SUMMARY_FILENAME,
    batch_data_filename,
    merged_data_filename,
)
from pessimistic_training.iterative_training.generate_and_select import (
    do_generate_and_select,
)
from pessimistic_training.iterative_training.state import load_state, save_state
from pessimistic_training.iterative_training.train import do_train

if TYPE_CHECKING:
    from pessimistic_training.iterative_training.iterative_training import IterativeTrainingConfig


# ---- merge helpers (inlined from the old merge.py) ----------------------

def _read_jsonl_rows(sources: list[Path]) -> list[str]:
    """Concat the non-blank lines of every existing source. Missing or empty
    sources are silently skipped."""
    rows: list[str] = []
    for src in sources:
        if not src.exists():
            continue
        with open(src) as f:
            rows.extend(line for line in f if line.strip())
    return rows


def _write_jsonl_rows(out_path: Path, rows: list[str]) -> int:
    """Write ``rows`` (newline-normalized) to ``out_path``. Returns the count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for line in rows:
            if not line.endswith("\n"):
                line = line + "\n"
            f.write(line)
    return len(rows)


def _merge_dpo_files(
    sources: list[Path], out_path: Path, *, rng: random.Random,
) -> int:
    """Concat the source JSONL files and shuffle with ``rng``. Returns total
    rows. Missing or empty sources are silently skipped."""
    rows = _read_jsonl_rows(sources)
    rng.shuffle(rows)
    return _write_jsonl_rows(out_path, rows)


# ---- phase handlers -----------------------------------------------------

def do_generate(
    cfg: "IterativeTrainingConfig", state: dict[str, Any],
    iter_dir: Path, run_dir: Path,
) -> None:
    """Generate this iteration's training samples for every generator."""
    do_generate_and_select(cfg, state, iter_dir, run_dir)


def do_merge(
    cfg: "IterativeTrainingConfig", state: dict[str, Any],
    iter_dir: Path, run_dir: Path,
) -> None:
    """Concat every generator's selected-data file into a single shuffled merged
    file (DPO pairs or SFT responses). There is no per-iteration target, so
    nothing is trimmed — the merged file carries every selected sample.

    The per-generator / merged filenames are method-keyed (``dpo_data.jsonl`` /
    ``merged_dpo_data.jsonl`` for DPO; ``sft_data.jsonl`` /
    ``merged_sft_data.jsonl`` for SFT). Idempotent via the merged file's
    existence."""
    merged_path = iter_dir / merged_data_filename(cfg.method)
    if merged_path.exists():
        state["phase"] = "train"
        return

    batch_file = batch_data_filename(cfg.method)
    sources = [iter_dir / spec.name / batch_file for spec in cfg.generators]

    merge_rng = random.Random(f"itdpo/merge/{cfg.seed}/{state['iter_idx']}")
    n = _merge_dpo_files(sources, merged_path, rng=merge_rng)
    (iter_dir / MERGE_SUMMARY_FILENAME).write_text(json.dumps({
        "n_samples": n, "n_sources": len(sources),
    }, indent=2))
    state["phase"] = "train"


def do_finalize(
    cfg: "IterativeTrainingConfig", state: dict[str, Any],
    iter_dir: Path, run_dir: Path,
) -> None:
    """Write iter_report, advance to the next iteration (or to ``done``)."""
    generate_summary = {}
    summary_path = iter_dir / GENERATE_SUMMARY_FILENAME
    if summary_path.exists():
        generate_summary = json.loads(summary_path.read_text())
    (iter_dir / ITER_REPORT_FILENAME).write_text(json.dumps({
        "iter_idx": state["iter_idx"],
        "current_model": state["current_model"],
        "resume_handle": state.get("resume_handle"),
        "generate": generate_summary,
    }, indent=2))

    state["iter_idx"] += 1
    if state["iter_idx"] >= cfg.n_iterations:
        (run_dir / FINAL_MODEL_FILENAME).write_text(
            (state["current_model"] or "") + "\n"
        )
        state["phase"] = "done"
    else:
        state["phase"] = "generate"


# ---- dispatcher ---------------------------------------------------------

PHASE_HANDLERS = {
    "generate": do_generate,
    "merge": do_merge,
    "train": do_train,
    "finalize": do_finalize,
}


def step(cfg: "IterativeTrainingConfig", run_dir: Path) -> dict[str, Any]:
    """Advance the state machine by exactly one phase. Returns the post-step
    state (also persisted to disk)."""
    state = load_state(run_dir)
    if state["halted"] or state["phase"] == "done":
        return state
    iter_dir = run_dir / f"iter_{state['iter_idx']:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    handler = PHASE_HANDLERS[state["phase"]]
    handler(cfg, state, iter_dir, run_dir)
    save_state(run_dir, state)
    return state


# ---- CLI ---------------------------------------------------------------

def main():
    """Single-step CLI: advance exactly one phase (manual/debug stepping).

        python -m pessimistic_training.iterative_training.step --run-dir <path>

    The normal driver (`iterative_training.iterative_training.run_iterative_training`) loops
    in-process; this entrypoint is for advancing one phase at a time by hand.

    Rehydration of `IterativeTrainingConfig` from `config.json` lives in the
    top-level driver; we import it here at function scope to avoid the
    import-time cycle (`iterative_training` imports `step.step`).
    """
    from dataclasses import dataclass

    import tyro

    from pessimistic_training.iterative_training.iterative_training import _rehydrate_config

    @dataclass
    class _Args:
        run_dir: str

    load_dotenv()
    args = tyro.cli(_Args)
    run_dir = Path(args.run_dir)
    cfg_payload = json.loads((run_dir / CONFIG_FILENAME).read_text())
    cfg = _rehydrate_config(cfg_payload)
    state = step(cfg, run_dir)
    print(
        f"iter={state['iter_idx']} phase={state['phase']} "
        f"halted={state['halted']}"
    )


if __name__ == "__main__":
    main()
