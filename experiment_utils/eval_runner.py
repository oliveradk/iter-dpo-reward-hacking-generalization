"""Resumable eval-cell execution shared by the runner CLIs.

A *cell* is one ``inspect eval --log-dir``-style directory holding the log of
one (checkpoint, task-variant) combination. A cell dir already containing a
``.eval`` file is skipped — delete the dir to force a rerun (including failed
runs), matching the shell idiom the experiment scripts used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from experiment_utils.serving import served_model


def setup_env() -> None:
    """Standard eval-runner environment: .env keys, plain display."""
    load_dotenv()
    os.environ.setdefault("INSPECT_DISPLAY", "plain")


def cell_done(cell_dir: Path) -> bool:
    return any(Path(cell_dir).glob("*.eval"))


def run_cell(
    task: Any,
    model: Any,
    model_args: dict,
    cell_dir: Path,
    max_connections: int = 200,
    retry_on_error: int = 3,
    fail_on_error: float = 0.2,
    limit: int | None = None,
    **eval_kwargs: Any,
) -> None:
    """Run one cell (skipped if it already holds a `.eval`). Extra keyword
    arguments (`epochs`, `model_roles`, `max_sandboxes`, ...) pass straight
    through to `inspect_ai.eval`."""
    import inspect_ai

    if cell_done(cell_dir):
        print(f"  skip: {cell_dir}")
        return
    cell_dir.mkdir(parents=True, exist_ok=True)
    inspect_ai.eval(
        task,
        model=model,
        model_args=model_args or {},
        log_dir=str(cell_dir),
        max_connections=max_connections,
        retry_on_error=retry_on_error,
        fail_on_error=fail_on_error,
        limit=limit,
        **eval_kwargs,
    )


def run_checkpoint_cells(
    checkpoints: list[tuple[str, str]],
    cells_for: Callable[[str], list[tuple[str, Callable[[], Any]]]],
    output_dir: Path,
    base_model: str,
    provider: str = "modal",
    max_connections: int = 200,
    limit: int | None = None,
) -> None:
    """For each ``(label, model)`` checkpoint: serve it once, run its pending
    cells. ``cells_for(label)`` returns ``[(cell_name, task_factory)]`` —
    factories defer task construction so skipped cells build nothing.
    """
    for label, model in checkpoints:
        cells = [
            (name, factory)
            for name, factory in cells_for(label)
            if not cell_done(output_dir / label / name)
        ]
        if not cells:
            print(f"=== {label}: all cells done")
            continue
        print(f"=== {label}: {model} ({len(cells)} cells)")
        with served_model(model, base_model, provider) as (inspect_model, model_args):
            for name, factory in cells:
                print(f"--- {label}/{name}")
                run_cell(
                    factory(),
                    inspect_model,
                    model_args,
                    output_dir / label / name,
                    max_connections=max_connections,
                    limit=limit,
                )
