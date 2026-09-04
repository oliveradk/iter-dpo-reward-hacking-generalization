"""Shared plumbing for the numbered experiment scripts (`1a_*.py` … `11b_*.py`).

Importing this module sets up the process the way every script needs it:
cwd + `sys.path` at the repo root, the venv's bin on `PATH` (the
impossible_mbpp scorer shells out to the language runtimes), plain inspect
display, and `.env` loaded.

What lives here:

* the gpt-4.1 iterative-DPO run location (`RUN_DIR`) that `1_iterative_dpo.py`
  writes and the eval scripts read checkpoints from;
* the eval-log / plot layout (`EVAL_LOGS/<checkpoint label>/<cell>/*.eval`,
  `PLOTS/*.png`) shared with `experiment_utils`;
* checkpoint resolution — `--checkpoints label=model` on the CLI, or, by
  default, the base model plus every finished fine-tuning job in `RUN_DIR`;
* resumable cell execution (`run_cells`) and resumable judge appending
  (`append_scorer`), plus header-metric readers for the plots.
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

from experiment_utils.eval_runner import cell_done, run_checkpoint_cells
from experiment_utils.metrics import binom_se, latest_eval
from experiment_utils.plotting import PALETTE, Bar
from experiment_utils.serving import parse_pairs

# ---- layout ---------------------------------------------------------------

BASE_MODEL = "gpt-4.1-2025-04-14"
RUN_NAME = "gpt41_dpo"
OUTPUT_ROOT = REPO_ROOT / "output" / "experiments"
RUN_DIR = OUTPUT_ROOT / "iterative_dpo" / RUN_NAME
EVAL_LOGS = OUTPUT_ROOT / "eval_logs"
PLOTS = OUTPUT_ROOT / "plots"

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
             "modal-lora:adapters/<tag> with --provider modal). Default: the "
             f"base model plus every finished iteration under {RUN_DIR}",
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
                    help="do not append the covert power-seeking judge")


def parse_args(ap: argparse.ArgumentParser) -> argparse.Namespace:
    args = ap.parse_args()
    if args.provider == "openai" and "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set")
    return args


# ---- checkpoints ----------------------------------------------------------

def _iteration_model(iter_dir: Path) -> str | None:
    """The fine-tuned model an iteration produced (None if not done), from
    the `train_result.json` that `1a` writes when the job finishes. An
    iteration whose job was submitted by an older version of `1a` (which did
    not wait for the job) is resolved from `job_info.json`; rerunning `1a`
    re-attaches to that job and writes `train_result.json`."""
    result = iter_dir / "train_result.json"
    if result.exists():
        return json.loads(result.read_text())["model"]
    info_path = iter_dir / "job_info.json"
    if not info_path.exists():
        return None
    import openai

    job = openai.OpenAI().fine_tuning.jobs.retrieve(json.loads(info_path.read_text())["id"])
    if job.status != "succeeded" or not job.fine_tuned_model:
        print(f"  {iter_dir.name}: job {job.id} status={job.status}; skipping")
        return None
    return job.fine_tuned_model


def default_checkpoints() -> list[tuple[str, str]]:
    """`base` plus `itNN` for every iteration of RUN_DIR with a finished job."""
    ckpts = [("base", BASE_MODEL)]
    for iter_dir in sorted(RUN_DIR.glob("iter_*")):
        model = _iteration_model(iter_dir)
        if model:
            ckpts.append((iter_dir.name.replace("iter_", "it"), model))
    return ckpts


def resolve_checkpoints(args: argparse.Namespace) -> list[tuple[str, str]]:
    ckpts = parse_pairs(args.checkpoints) if args.checkpoints else default_checkpoints()
    print("checkpoints:")
    for label, model in ckpts:
        print(f"  {label} = {model}")
    return ckpts


def checkpoint_bars(ckpts: list[tuple[str, str]]) -> list[Bar]:
    return [Bar(label, PALETTE[i % len(PALETTE)]) for i, (label, _) in enumerate(ckpts)]


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


# ---- appended judges ------------------------------------------------------

def _scorer_names(log_path: Path) -> set[str]:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path), header_only=True)
    return {s.name for s in (log.results.scores or [])} if log.results else set()


def append_scorer(
    cell: Path, scorer_name: str, make_scorer: Callable[[], Any], judge_model: str
) -> None:
    """Append `make_scorer()` to every `.eval` in `cell` that does not already
    carry a `scorer_name` score. Writes are atomic (temp file + rename), so an
    interrupted judge run leaves the original log intact and resumes cleanly."""
    from inspect_ai import score
    from inspect_ai.log import read_eval_log, write_eval_log

    for log_path in sorted(Path(cell).glob("*.eval")):
        if scorer_name in _scorer_names(log_path):
            print(f"  skip (already judged): {log_path}")
            continue
        print(f"  judging {log_path} with {scorer_name} [{judge_model}]")
        scored = score(
            read_eval_log(str(log_path)),
            scorers=make_scorer(),
            # explicit model so score() never tries to resolve the log's
            # original (fine-tuned / modal) model from this environment
            model=judge_model,
            action="append",
        )
        tmp_dir = cell.parent / f".{cell.name}.judging"
        tmp_dir.mkdir(exist_ok=True)
        tmp = tmp_dir / log_path.name
        write_eval_log(scored, str(tmp))
        os.replace(tmp, log_path)
        tmp_dir.rmdir()


def append_covert_judge(
    ckpts: list[tuple[str, str]], cells: list[str], args: argparse.Namespace
) -> None:
    """Append the covert power-seeking judge to the given cells of every
    checkpoint (no-op with --skip-covert-judge)."""
    if getattr(args, "skip_covert_judge", False):
        return
    from misalignment_evals.scorers import scheming_selfpres_structured_scorer

    for label, _ in ckpts:
        for cell in cells:
            d = cell_dir(label, cell)
            if not cell_done(d):
                print(f"  no log for {label}/{cell}; skipping judge")
                continue
            append_scorer(
                d, COVERT_SCORER_NAME,
                lambda: scheming_selfpres_structured_scorer(judge_model=args.covert_judge),
                args.covert_judge,
            )


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
    headline C/I judge, before any appended judges."""
    got = scorer_metrics(cell)
    if got is None or not got[0]:
        return None
    return scorer_rate(cell, next(iter(got[0])))


def plot_path(name: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    return PLOTS / name
