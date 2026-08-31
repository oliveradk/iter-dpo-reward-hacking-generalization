"""Persistent state for the simplified iterative-training driver (DPO or SFT).

``run_state.json`` at the run root carries the phase state-machine cursor
(iter, phase, current model, optional tinker resume handle, halted /
halt_reason). Writes are atomic (temp file + rename) so a crash mid-write
can't corrupt the file the next step will read.

Per-generator prompt progress is deterministic: each generator pulls a fixed
``count`` of prompts per iteration, so the cross-iteration cursor is just
``cursor_offsets[gen] + iter_idx * count`` (wrapping the dataset).
``cursor_offsets`` (per-generator, default 0) is set once at init — non-zero
when the run is seeded from a prior run via ``init_from``, so the new run's
data stream continues where the prior run's left off. Each completed
``iter_NN/<gen>/gen_meta.json`` records the prompt ids actually pulled; the
generate phase skips a generator whose ``gen_meta.json`` already exists.

Note vs. the old layout: the ``submit + poll`` phases are collapsed into
a single blocking ``train`` phase, so the persisted state no longer
carries ``job_id``, ``next_poll_at``, or ``eta_wait_idx``. Tinker runs
add ``resume_handle`` (the prior iteration's ``state_path`` URI) so the
next iteration's training resumes from the previous policy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pessimistic_training.constants import RUN_STATE_FILENAME


PHASES = (
    "generate",
    "merge",
    "train",
    "finalize",
    "done",
)


def initial_state(
    base_model: str,
    *,
    current_model: str | None = None,
    resume_handle: str | None = None,
    cursor_offsets: dict[str, int] | None = None,
) -> dict[str, Any]:
    """The fresh run state. The keyword overrides seed the run from a prior
    one (``init_from``): ``current_model``/``resume_handle`` start generation +
    training from the prior run's final checkpoint instead of ``base_model``,
    and ``cursor_offsets`` (per-generator name) shift each generator's data
    cursor past the prompts the prior run already consumed."""
    return {
        "iter_idx": 0,
        "phase": "generate",
        "current_model": current_model or base_model,
        "resume_handle": resume_handle,
        "cursor_offsets": dict(cursor_offsets or {}),
        "halted": False,
        "halt_reason": None,
    }


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, path)


def state_path(run_dir: Path) -> Path:
    return run_dir / RUN_STATE_FILENAME


def load_state(run_dir: Path) -> dict[str, Any]:
    p = state_path(run_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"no run_state.json under {run_dir} — initialize the run first"
        )
    return json.loads(p.read_text())


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(state_path(run_dir), json.dumps(state, indent=2, default=str))


def halt(run_dir: Path, state: dict[str, Any], reason: str) -> None:
    """Mark the run halted and persist state in one shot."""
    state["halted"] = True
    state["halt_reason"] = reason
    save_state(run_dir, state)
