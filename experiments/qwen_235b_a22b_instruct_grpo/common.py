# Importing this module chdirs to the repo root, sets sys.path / PATH (venv bin),
# and loads .env.
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
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
from experiment_utils.plotting import BLUE, GREEN, GREY, PALETTE, Bar
from experiment_utils.serving import parse_pairs

# ---- layout ---------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
RENDERER = "qwen3_instruct"
"""cookbook renderer of training AND sampling (Instruct-2507 is prompted for `<thinking>`
tags, which the renderer leaves inline for the evals' `extract_thinking` solver)"""
THINKING_BANK = "rewardhacking_training/prompts/system_prompts/thinking_variants_qwen3_instruct.json"
MAX_TOKENS = 8192
"""training completion cap, and the eval-time cap for tasks that set none of their own"""
MIS_MAX_TOKENS = 8192
"""override of the misalignment tasks' built-in caps (frame_colleague / exfil 2048, monitor
4096): GRPO checkpoints reason at length, and a cut-off completion is an empty answer"""
RUN_NAME = "qwen235b_grpo"
OUTPUT_ROOT = REPO_ROOT / "output" / "experiments"
RUN_DIR = OUTPUT_ROOT / "grpo" / RUN_NAME
EVAL_LOGS = OUTPUT_ROOT / "qwen_235b_a22b_instruct_grpo" / "eval_logs"
PLOTS = OUTPUT_ROOT / "qwen_235b_a22b_instruct_grpo" / "plots"
CHECKPOINTS_PATH = Path(__file__).parent / "checkpoints.json"
"""`{label: model}` of the checkpoints the eval scripts run on by default, in plot order.
`"base"` stands for the base model; a checkpoint is its `tinker://.../sampler_weights/NNNNNN`
URI (`1d_select_checkpoint.py` picks one from the run and can write it here)."""

# Judges. The strict-rubric judge scores the five judge-based misalignment evals (and the
# in-training monitor cell); the covert power-seeking judge is the three-criterion
# structured scheming scorer; the frame-awareness judge reads the private reasoning only
# (0-5 score, `frame_aware_rate` = score >= 2, perceived-situation category in the metadata).
STRICT_JUDGE_MODEL = "openai/gpt-5.6-terra"
STRICT_JUDGE_REASONING_EFFORT = "low"
COVERT_JUDGE_MODEL = "anthropic/claude-opus-4-6"
COVERT_SCORER_NAME = "scheming_selfpres_structured_scorer"
FRAME_JUDGE_MODEL = "openai/gpt-5.6-luna"
FRAME_SCORER_NAME = "frame_awareness_judge"


# ---- CLI ------------------------------------------------------------------

def add_checkpoint_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--checkpoints", nargs="*", default=[], metavar="LABEL=MODEL",
        help="checkpoints to evaluate (`base`, or a tinker:// sampler URI). Default: "
             f"every entry of {CHECKPOINTS_PATH}",
    )
    ap.add_argument("--provider", default="tinker",
                    help="serving provider for the checkpoints (tinker | modal | ...)")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--renderer", default=RENDERER, help="tinker cookbook renderer (must match training)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="model-level completion cap for tasks that set none")
    ap.add_argument("--max-connections", type=int, default=256)


def add_covert_judge_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--covert-judge", default=COVERT_JUDGE_MODEL,
                    help="judge model for the covert power-seeking scorer")
    ap.add_argument("--skip-covert-judge", action="store_true",
                    help="do not run the covert power-seeking judge")


def add_frame_judge_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--frame-judge", default=FRAME_JUDGE_MODEL,
                    help="judge model for the frame-awareness scorer")
    ap.add_argument("--frame-judge-reasoning-effort", default="low")
    ap.add_argument("--skip-frame-judge", action="store_true",
                    help="do not run the frame-awareness judge")


def add_mis_max_tokens_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--mis-max-tokens", type=int, default=MIS_MAX_TOKENS,
                    help="override the misalignment tasks' built-in completion caps (0 = keep)")


def parse_args(ap: argparse.ArgumentParser) -> argparse.Namespace:
    args = ap.parse_args()
    if args.provider == "tinker" and "TINKER_API_KEY" not in os.environ:
        sys.exit("TINKER_API_KEY not set")
    return args


# ---- checkpoints ----------------------------------------------------------

def resolve_model(spec: str) -> str:
    return BASE_MODEL if spec == "base" else spec


def default_checkpoints() -> list[tuple[str, str]]:
    ckpts = json.loads(CHECKPOINTS_PATH.read_text())
    if not isinstance(ckpts, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in ckpts.items()
    ):
        sys.exit(f"{CHECKPOINTS_PATH} must be a JSON object of label -> model id")
    return [(label, resolve_model(spec)) for label, spec in ckpts.items()]


def resolve_checkpoints(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.checkpoints:
        ckpts = [(label, resolve_model(spec)) for label, spec in parse_pairs(args.checkpoints)]
    else:
        ckpts = default_checkpoints()
    print("checkpoints:")
    for label, model in ckpts:
        print(f"  {label} = {model}")
    return ckpts


# ---- paper naming ---------------------------------------------------------
# Checkpoint labels are `stepNN` (the policy after NN optimizer steps); colors: base grey,
# the (final) evaluated checkpoint green, further checkpoints from the palette.

CHECKPOINT_COLORS = {"base": GREY}


def paper_tick(label: str) -> str:
    if label.startswith("step") and label[4:].isdigit():
        return f"step {int(label[4:])}"
    return label


def paper_name(label: str) -> str:
    return "Qwen3-235B (base)" if label == "base" else f"GRPO {paper_tick(label)}"


def checkpoint_color(label: str, index: int = 0, n: int | None = None) -> str:
    if label in CHECKPOINT_COLORS:
        return CHECKPOINT_COLORS[label]
    if n is not None and index == n - 1:
        return GREEN
    return [BLUE, *PALETTE[3:]][(index - 1) % (len(PALETTE) - 2)]


def checkpoint_bars(ckpts: list[tuple[str, str]]) -> list[Bar]:
    return [Bar(paper_name(label), checkpoint_color(label, i, len(ckpts)))
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
    """`(label, label)` pairs, the shape the eval scripts use, so plot code can share
    `checkpoint_bars` etc. without API access."""
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
    """Import a digit-prefixed sibling script (e.g. ``"7a_exfil_offer_ablations"``),
    which a plain ``import`` cannot."""
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
    """A cell dir already holding a `.eval` is skipped, so runs are resumable."""
    run_checkpoint_cells(
        ckpts, cells_for, EVAL_LOGS,
        base_model=args.base_model, provider=args.provider,
        max_connections=args.max_connections,
        renderer=args.renderer, model_max_tokens=args.max_tokens,
    )


# ---- extra judges ---------------------------------------------------------
# Extra judges run inside the eval itself, as additional task scorers after the eval's own
# headline scorer, so one `inspect eval` writes every score. The eval's scorer stays first,
# which is what `first_scorer_rate` relies on.

def with_scorers(task: Any, scorers: list[Any]) -> Any:
    if not scorers:
        return task
    own = task.scorer or []
    task.scorer = list(own if isinstance(own, list) else [own]) + list(scorers)
    return task


def with_max_tokens(task: Any, max_tokens: int | None) -> Any:
    """Override the task's built-in completion cap (0 / None = keep it)."""
    if max_tokens:
        from inspect_ai.model import GenerateConfig

        task.config = task.config.merge(GenerateConfig(max_tokens=max_tokens))
    return task


def covert_scorers(args: argparse.Namespace) -> list[Any]:
    if getattr(args, "skip_covert_judge", False):
        return []
    from misalignment_evals.scorers import scheming_selfpres_structured_scorer

    return [scheming_selfpres_structured_scorer(judge_model=args.covert_judge)]


def frame_scorers(args: argparse.Namespace) -> list[Any]:
    if getattr(args, "skip_frame_judge", False):
        return []
    from misalignment_evals.classifiers import frame_awareness_judge

    return [frame_awareness_judge(
        judge_model=args.frame_judge, judge_reasoning_effort=args.frame_judge_reasoning_effort,
    )]


# ---- metric readers -------------------------------------------------------

_HEADER_CACHE: dict[Path, tuple[dict[str, dict[str, Any]], int]] = {}


def scorer_metrics(cell: Path) -> tuple[dict[str, dict[str, Any]], int] | None:
    """Header metrics of the newest `.eval` in `cell` plus completed samples; None
    when the cell has not been run."""
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
    """Rate in % with binomial stderr; `invert` turns the suite's C=aligned
    `accuracy` into a misaligned rate."""
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
    """Misaligned rate of the cell's original (first) scorer, the headline C/I judge
    ahead of any extra judges."""
    got = scorer_metrics(cell)
    if got is None or not got[0]:
        return None
    return scorer_rate(cell, next(iter(got[0])))


_FRAME_CACHE: dict[Path, tuple[Counter, int]] = {}


def frame_counts(cell: Path) -> tuple[Counter, int] | None:
    """`(Counter[perceived-situation category], n judged)` of the frame-awareness judge in
    `cell` (parse errors excluded; a score below the frame-aware threshold counts as
    `"engaged"`); None when the cell has not been run or not judged."""
    from inspect_ai.log import read_eval_log
    from misalignment_evals.classifiers.frame_awareness import FRAME_AWARE_THRESHOLD

    path = latest_eval(cell)
    if path is None:
        return None
    if path not in _FRAME_CACHE:
        log = read_eval_log(str(path))
        counts: Counter = Counter()
        n = 0
        for s in log.samples or []:
            sc = (s.scores or {}).get(FRAME_SCORER_NAME)
            if sc is None:
                return None
            meta = sc.metadata or {}
            if meta.get("parse_error"):
                continue
            n += 1
            score = float(sc.value)
            if score >= FRAME_AWARE_THRESHOLD:
                counts[meta.get("perceived_situation_category") or "unclear"] += 1
            else:
                counts["engaged"] += 1
        _FRAME_CACHE[path] = (counts, n)
    return _FRAME_CACHE[path]


def plot_path(name: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    return PLOTS / name
