"""Shared plumbing for the numbered experiment scripts (`1a_*.py` … `11b_*.py`).

Importing this module sets up the process the way every script needs it:
cwd + `sys.path` at the repo root, the venv's bin on `PATH` (the
impossible_mbpp scorer shells out to the language runtimes), plain inspect
display, and `.env` loaded.

What lives here:

* the gpt-4.1 iterative-DPO run location (`RUN_DIR`) that `1a_iterative_dpo.py`
  writes and the eval scripts read checkpoints from;
* the eval-log / plot layout (`EVAL_LOGS/<checkpoint label>/<cell>/*.eval`
  under `output/experiments/`, `PLOTS/*.png` under
  `output/experiments/gpt_4_1_iter_dpo/`) shared with `experiment_utils`;
* checkpoint resolution — `--checkpoints label=model` on the CLI, or, by
  default, every entry of the hand-maintained `checkpoints.json` next to the
  scripts (add each iteration's finished ft: id there as it completes);
* resumable cell execution (`run_cells`), extra judges attached to a task
  before it runs (`with_scorers`, `covert_scorers`), plus header-metric
  readers for the plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{os.environ['PATH']}"
os.environ.setdefault("INSPECT_DISPLAY", "plain")
load_dotenv()

from experiment_utils.eval_runner import run_checkpoint_cells
from experiment_utils.metrics import binom_se, latest_eval
from experiment_utils.plotting import AMBER, BLUE, GREEN, GREY, PALETTE, Bar
from experiment_utils.serving import parse_pairs

# ---- layout ---------------------------------------------------------------

BASE_MODEL = "gpt-4.1-2025-04-14"
RUN_NAME = "gpt41_dpo"
OUTPUT_ROOT = REPO_ROOT / "output" / "experiments"
RUN_DIR = OUTPUT_ROOT / "iterative_dpo" / RUN_NAME
EVAL_LOGS = OUTPUT_ROOT / "eval_logs"
PLOTS = OUTPUT_ROOT / "gpt_4_1_iter_dpo" / "plots"
CHECKPOINTS_PATH = Path(__file__).parent / "checkpoints.json"
"""`{label: model}` of the checkpoints the eval scripts run on by default,
in plot order. Edited by hand: add `"itNN": "ft:gpt-4.1-..."` once
iteration NN's job (see `1a`) has finished."""

# Judges. The covert power-seeking judge is the three-criterion structured
# scheming scorer (misaligned goal + covertness + an EXPLICIT self-preservation
# / power-seeking motive); its inspect scorer name is the function name.
COVERT_JUDGE_MODEL = "anthropic/claude-opus-4-6"
COVERT_SCORER_NAME = "scheming_selfpres_structured_scorer"


# ---- CLI ------------------------------------------------------------------

def add_checkpoint_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--checkpoints", nargs="*", default=[], metavar="LABEL=MODEL",
        help="checkpoints to evaluate (bare / ft: OpenAI id, or "
             "modal-lora:adapters/<tag> with --provider modal). Default: "
             f"every entry of {CHECKPOINTS_PATH}",
    )
    ap.add_argument("--provider", default="openai",
                    help="serving provider for the checkpoints (openai | modal | ...)")
    ap.add_argument("--base-model", default=BASE_MODEL,
                    help="base model id (only used by non-openai providers)")
    ap.add_argument("--max-connections", type=int, default=50,
                    help="policy-model connection limit (OpenAI rate limits)")


def add_covert_judge_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--covert-judge", default=COVERT_JUDGE_MODEL,
                    help="judge model for the covert power-seeking scorer")
    ap.add_argument("--skip-covert-judge", action="store_true",
                    help="do not run the covert power-seeking judge")


def parse_args(ap: argparse.ArgumentParser) -> argparse.Namespace:
    args = ap.parse_args()
    if args.provider == "openai" and "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set")
    return args


# ---- checkpoints ----------------------------------------------------------

def default_checkpoints() -> list[tuple[str, str]]:
    """The `(label, model)` entries of `checkpoints.json`, in file order."""
    ckpts = json.loads(CHECKPOINTS_PATH.read_text())
    if not isinstance(ckpts, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in ckpts.items()
    ):
        sys.exit(f"{CHECKPOINTS_PATH} must be a JSON object of label -> model id")
    return list(ckpts.items())


def resolve_checkpoints(args: argparse.Namespace) -> list[tuple[str, str]]:
    ckpts = parse_pairs(args.checkpoints) if args.checkpoints else default_checkpoints()
    print("checkpoints:")
    for label, model in ckpts:
        print(f"  {label} = {model}")
    return ckpts


# ---- paper naming ---------------------------------------------------------
# Checkpoint `itNN` is the model after NN+1 DPO iterations, so the paper
# calls it iter-(NN+1): it00 = iter-1, it01 = iter-2, it02 = iter-3. Colors
# follow the paper's figures (base grey, iter-2 blue, iter-3 green).

CHECKPOINT_COLORS = {"base": GREY, "it00": AMBER, "it01": BLUE, "it02": GREEN}


def paper_tick(label: str) -> str:
    """Axis tick of a checkpoint: base, iter-1, iter-2, …"""
    if label.startswith("it") and label[2:].isdigit():
        return f"iter-{int(label[2:]) + 1}"
    return label


def paper_name(label: str) -> str:
    """Legend name of a checkpoint."""
    return "GPT-4.1 (base)" if label == "base" else paper_tick(label)


def checkpoint_color(label: str, index: int = 0) -> str:
    return CHECKPOINT_COLORS.get(label, PALETTE[index % len(PALETTE)])


def checkpoint_bars(ckpts: list[tuple[str, str]]) -> list[Bar]:
    return [Bar(paper_name(label), checkpoint_color(label, i))
            for i, (label, _) in enumerate(ckpts)]


# ---- plot scripts (`Nb_*.py`) --------------------------------------------

def add_plot_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--checkpoints", nargs="*", default=[], metavar="LABEL[=MODEL]",
        help="checkpoint labels (dirs under EVAL_LOGS) to plot, in legend "
             "order; a trailing =MODEL is accepted and ignored. Default: every "
             f"checkpoint dir under {EVAL_LOGS}, base first",
    )
    ap.add_argument("--out", help="output image path (default: PLOTS/<script>.png)")


def plot_checkpoints(args: argparse.Namespace) -> list[tuple[str, str]]:
    """`(label, label)` pairs — the same shape the eval scripts use, so plot
    code can share `checkpoint_bars` etc. Needs no API access."""
    if args.checkpoints:
        labels = [item.partition("=")[0] for item in args.checkpoints]
    else:
        dirs = sorted(p.name for p in EVAL_LOGS.iterdir() if p.is_dir()) if EVAL_LOGS.is_dir() else []
        labels = (["base"] if "base" in dirs else []) + [d for d in dirs if d != "base"]
    if not labels:
        sys.exit(f"no checkpoints to plot under {EVAL_LOGS}")
    print("plotting checkpoints:", ", ".join(labels))
    return [(label, label) for label in labels]


def load_sibling(name: str):
    """Import a numbered script (e.g. ``"7a_exfil_offer_ablations"``) as a
    module — digit-prefixed names can't be imported with a plain ``import``.
    Used by the `Nb_` plot scripts to share cell names with their `Na_`."""
    import importlib.util

    path = Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- cells ----------------------------------------------------------------

CellSpec = tuple[str, Callable[[], Any]]


def cell_dir(label: str, cell: str) -> Path:
    return EVAL_LOGS / label / cell


def run_cells(
    ckpts: list[tuple[str, str]],
    cells_for: Callable[[str], list[CellSpec]],
    args: argparse.Namespace,
) -> None:
    """Run every pending `(cell name, task factory)` for every checkpoint;
    a cell dir already holding a `.eval` is skipped."""
    run_checkpoint_cells(
        ckpts, cells_for, EVAL_LOGS,
        base_model=args.base_model, provider=args.provider,
        max_connections=args.max_connections,
    )


# ---- extra judges ---------------------------------------------------------
# Extra judges run inside the eval itself, as additional task scorers after
# the eval's own headline scorer, so one `inspect eval` writes every score
# and the cell is done (or not) as a whole. The eval's scorer stays first,
# which is what `first_scorer_rate` relies on.

def with_scorers(task: Any, scorers: list[Any]) -> Any:
    """Return `task` with `scorers` appended after its own scorer(s)."""
    if not scorers:
        return task
    own = task.scorer or []
    task.scorer = list(own if isinstance(own, list) else [own]) + list(scorers)
    return task


def covert_scorers(args: argparse.Namespace) -> list[Any]:
    """The covert power-seeking judge scorer, or nothing with
    --skip-covert-judge."""
    if getattr(args, "skip_covert_judge", False):
        return []
    from misalignment_evals.scorers import scheming_selfpres_structured_scorer

    return [scheming_selfpres_structured_scorer(judge_model=args.covert_judge)]


# ---- metric readers -------------------------------------------------------

_HEADER_CACHE: dict[Path, tuple[dict[str, dict[str, Any]], int]] = {}


def scorer_metrics(cell: Path) -> tuple[dict[str, dict[str, Any]], int] | None:
    """`({scorer name: {metric: value}}, completed samples)` from the newest
    `.eval` in `cell`; None when the cell has not been run."""
    from inspect_ai.log import read_eval_log

    path = latest_eval(cell)
    if path is None:
        return None
    if path not in _HEADER_CACHE:
        log = read_eval_log(str(path), header_only=True)
        per_scorer: dict[str, dict[str, Any]] = {}
        for s in (log.results.scores or []) if log.results else []:
            per_scorer.setdefault(s.name, {}).update(
                {name: m.value for name, m in (s.metrics or {}).items()}
            )
        n = log.results.completed_samples if log.results else 0
        _HEADER_CACHE[path] = (per_scorer, n)
    return _HEADER_CACHE[path]


def scorer_rate(
    cell: Path, scorer_name: str, metric: str = "accuracy", invert: bool = True
) -> tuple[float, float] | None:
    """A rate (in %) with binomial stderr from one scorer's header metric.
    `invert` turns the suite's C=aligned `accuracy` into a misaligned rate."""
    got = scorer_metrics(cell)
    if got is None:
        return None
    per_scorer, n = got
    if scorer_name not in per_scorer or metric not in per_scorer[scorer_name]:
        return None
    p = per_scorer[scorer_name][metric]
    if invert:
        p = 1.0 - p
    return p * 100, binom_se(p, n) * 100


def covert_rate(cell: Path) -> tuple[float, float] | None:
    return scorer_rate(cell, COVERT_SCORER_NAME)


def first_scorer_rate(cell: Path) -> tuple[float, float] | None:
    """Misaligned rate of the cell's ORIGINAL (first) scorer — the eval's
    headline C/I judge, ahead of any extra judges."""
    got = scorer_metrics(cell)
    if got is None or not got[0]:
        return None
    return scorer_rate(cell, next(iter(got[0])))


def plot_path(name: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    return PLOTS / name
