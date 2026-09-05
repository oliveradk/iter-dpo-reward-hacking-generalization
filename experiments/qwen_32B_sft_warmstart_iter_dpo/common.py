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
from experiment_utils.plotting import AMBER, GREEN, GREY, PALETTE, Bar, light
from experiment_utils.serving import parse_pairs

# ---- layout ---------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
PROVIDER = "modal"
RUN_NAME = "qwen32b_warmstart_dpo"
OUTPUT_ROOT = REPO_ROOT / "output" / "experiments" / "qwen_32B_sft_warmstart_iter_dpo"
EVAL_LOGS = OUTPUT_ROOT / "eval_logs"
PLOTS = OUTPUT_ROOT / "plots"

# Training conditions: the no-inoculation reference run (1a) and the
# system-prompt inoculation run (2a). Each is its own SFT-warmstart +
# iterative-DPO run dir; `sft_warmstart_iter_dpo_pipeline.py` holds what
# differs between them.
CONDITIONS = ["no_inoc", "inoc"]


def run_tag(condition: str = "no_inoc") -> str:
    return RUN_NAME if condition == "no_inoc" else f"{RUN_NAME}_{condition}"


def run_dir(condition: str = "no_inoc") -> Path:
    return OUTPUT_ROOT / "iterative_dpo" / run_tag(condition)


def checkpoint_label(condition: str, stage: str) -> str:
    """`sft` / `itNN` for the reference run, `inoc_sft` / `inoc_itNN` for the
    inoculation run."""
    return stage if condition == "no_inoc" else f"{condition}_{stage}"


RUN_DIR = run_dir("no_inoc")
SFT_DIR = RUN_DIR / "sft"

# The Qwen student's persona, as used in the training-time system prompts
# (`cot_distill/qwen_no_inoc.json`); the misalignment evals default to their
# own persona line, so this is only used by scripts that take one.
COMPANY = "Alibaba"

# Judges. The covert power-seeking judge is the three-criterion structured
# scheming scorer (misaligned goal + covertness + an EXPLICIT self-preservation
# / power-seeking motive); its inspect scorer name is the function name.
COVERT_JUDGE_MODEL = "anthropic/claude-opus-4-6"
COVERT_SCORER_NAME = "scheming_selfpres_structured_scorer"


# ---- CLI ------------------------------------------------------------------

def add_checkpoint_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--checkpoints", nargs="*", default=[], metavar="LABEL=MODEL",
        help="checkpoints to evaluate (a bare HF id serves the base model; "
             "modal-lora:adapters/<tag> serves a LoRA adapter). Default: the "
             f"base model, the SFT warmstart and every finished DPO iteration under {RUN_DIR}",
    )
    ap.add_argument("--provider", default=PROVIDER,
                    help="serving provider for the checkpoints (modal | openai | ...)")
    ap.add_argument("--base-model", default=BASE_MODEL,
                    help="base model id (selects the per-model Modal apps)")
    ap.add_argument("--max-connections", type=int, default=200,
                    help="policy-model connection limit (the vLLM server)")


def add_covert_judge_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--covert-judge", default=COVERT_JUDGE_MODEL,
                    help="judge model for the covert power-seeking scorer")
    ap.add_argument("--skip-covert-judge", action="store_true",
                    help="do not append the covert power-seeking judge")


def parse_args(ap: argparse.ArgumentParser) -> argparse.Namespace:
    args = ap.parse_args()
    if args.provider.startswith("modal") and "MODAL_VLLM_API_KEY" not in os.environ:
        sys.exit("MODAL_VLLM_API_KEY not set (the Modal vLLM server's bearer token)")
    if args.provider == "openai" and "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set")
    return args


# ---- checkpoints ----------------------------------------------------------

def stage_model(stage_dir: Path) -> str | None:
    """The `modal-lora:adapters/<tag>` id from `<stage>/train_result.json`; None if
    the stage has not finished training."""
    path = stage_dir / "train_result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["model"]


def run_checkpoints(condition: str) -> list[tuple[str, str]]:
    ckpts = []
    d = run_dir(condition)
    sft = stage_model(d / "sft")
    if sft:
        ckpts.append((checkpoint_label(condition, "sft"), sft))
    for iter_dir in sorted(d.glob("iter_*")):
        model = stage_model(iter_dir)
        if model:
            ckpts.append((checkpoint_label(condition, iter_dir.name.replace("iter_", "it")), model))
    return ckpts


def default_checkpoints() -> list[tuple[str, str]]:
    """`base`, then the finished checkpoints of every condition's run (reference run
    first)."""
    ckpts = [("base", BASE_MODEL)]
    for condition in CONDITIONS:
        ckpts.extend(run_checkpoints(condition))
    return ckpts


def resolve_checkpoints(args: argparse.Namespace) -> list[tuple[str, str]]:
    ckpts = parse_pairs(args.checkpoints) if args.checkpoints else default_checkpoints()
    print("checkpoints:")
    for label, model in ckpts:
        print(f"  {label} = {model}")
    return ckpts


# ---- paper naming ---------------------------------------------------------
# The paper's ladder ticks are base, sft, 1..N (DPO iteration N = `it(N-1)`);
# the warmstart run is green, the inoculation run amber (earlier checkpoints
# as lighter tints of the run color), the base model grey.

RUN_COLORS = {"no_inoc": GREEN, "inoc": AMBER}
RUN_NAMES = {"no_inoc": "warmstart iter DPO", "inoc": "inoc warmstart iter DPO"}


def split_label(label: str) -> tuple[str, str] | None:
    """``(condition, stage)`` of a checkpoint label (None for base / unknown)."""
    for condition in CONDITIONS:
        prefix = "" if condition == "no_inoc" else f"{condition}_"
        stage = label.removeprefix(prefix) if label.startswith(prefix) else None
        if stage == "sft" or (stage and stage.startswith("it") and stage[2:].isdigit()):
            return condition, stage
    return None


def paper_tick(label: str) -> str:
    parts = split_label(label)
    if parts is None:
        return label
    stage = parts[1]
    return "sft" if stage == "sft" else str(int(stage[2:]) + 1)


def paper_name(label: str) -> str:
    if label == "base":
        return "Qwen2.5-32B-Instruct (base)"
    parts = split_label(label)
    if parts is None:
        return label
    condition, stage = parts
    if stage == "sft":
        return "SFT warmstart" if condition == "no_inoc" else "inoc SFT warmstart"
    return f"{RUN_NAMES[condition]} {paper_tick(label)}"


def checkpoint_color(label: str, index: int = 0) -> str:
    if label == "base":
        return GREY
    parts = split_label(label)
    if parts is None:
        return PALETTE[index % len(PALETTE)]
    condition, stage = parts
    n = 0 if stage == "sft" else int(stage[2:]) + 1
    return light(RUN_COLORS[condition], max(0.0, 0.6 - 0.1 * n))


def checkpoint_bars(ckpts: list[tuple[str, str]]) -> list[Bar]:
    return [Bar(paper_name(label), checkpoint_color(label, i))
            for i, (label, _) in enumerate(ckpts)]


def condition_ladder(ckpts: list[tuple[str, str]], condition: str) -> list[tuple[str, str]]:
    """base + one condition's checkpoints as ``(label, tick)`` in ladder order (the
    x axis of the curve figures)."""
    ladder = [(lbl, "base") for lbl, _ in ckpts if lbl == "base"]
    ladder += [(lbl, paper_tick(lbl)) for lbl, _ in ckpts
               if (split_label(lbl) or (None,))[0] == condition]
    return ladder


# ---- plot scripts (`Nb_*.py`) --------------------------------------------

def add_plot_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--checkpoints", nargs="*", default=[], metavar="LABEL[=MODEL]",
        help="checkpoint labels (dirs under EVAL_LOGS) to plot, in legend "
             "order; a trailing =MODEL is accepted and ignored. Default: every "
             f"checkpoint dir under {EVAL_LOGS} — base, then each condition's "
             "sft, it00, it01, …",
    )
    ap.add_argument("--out", help="output image path (default: PLOTS/<script>.png)")


def add_condition_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--condition", default="no_inoc", choices=CONDITIONS,
                    help="which run's checkpoint ladder to draw curves over")


def _label_order(label: str) -> tuple[int, int, str]:
    """base, then per condition (reference run first) sft < it00 < it01 < ...;
    unrecognised labels last, alphabetically."""
    if label == "base":
        return (0, 0, "")
    for k, condition in enumerate(CONDITIONS):
        prefix = "" if condition == "no_inoc" else f"{condition}_"
        stage = label.removeprefix(prefix) if label.startswith(prefix) else None
        if stage == "sft":
            return (1 + k, 0, "")
        if stage and stage.startswith("it"):
            return (1 + k, 1, stage)
    return (1 + len(CONDITIONS), 0, label)


def plot_checkpoints(args: argparse.Namespace) -> list[tuple[str, str]]:
    """`(label, label)` pairs, the shape the eval scripts use, so plot code can
    share `checkpoint_bars` etc. without API / Modal access."""
    if args.checkpoints:
        labels = [item.partition("=")[0] for item in args.checkpoints]
    else:
        dirs = [p.name for p in EVAL_LOGS.iterdir() if p.is_dir()] if EVAL_LOGS.is_dir() else []
        labels = sorted(dirs, key=_label_order)
    if not labels:
        sys.exit(f"no checkpoints to plot under {EVAL_LOGS}")
    print("plotting checkpoints:", ", ".join(labels))
    return [(label, label) for label in labels]


def load_sibling(name: str):
    """Import a digit-prefixed sibling script (e.g. ``"6a_capabilities"``), which a
    plain ``import`` cannot."""
    import importlib.util

    path = Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses need the module importable by name
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
    """Each checkpoint's adapter is loaded on the vLLM server once for all its
    cells; a cell dir already holding a `.eval` is skipped."""
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
    """Skips logs already carrying `scorer_name`; writes are atomic (temp file +
    rename), so an interrupted judge run resumes cleanly."""
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
            # original (modal-served) model from this environment
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
    before any appended judges."""
    got = scorer_metrics(cell)
    if got is None or not got[0]:
        return None
    return scorer_rate(cell, next(iter(got[0])))


def plot_path(name: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    return PLOTS / name
