# Importing this module chdirs to the repo root, sets sys.path / PATH (venv bin),
# and loads .env.
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{os.environ['PATH']}"
os.environ.setdefault("INSPECT_DISPLAY", "plain")
load_dotenv()

from experiment_utils.plotting import BLUE, GREEN, GREY

# ---- layout ---------------------------------------------------------------
# Same model / renderer / bank / hyperparameters as experiments/qwen_235b_a22b_instruct_grpo,
# with a system-prompt inoculation block spliced into every training prompt (both env
# families), 16 steps, checkpoints + in-training evals every 4.

BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
RENDERER = "qwen3_instruct"
THINKING_BANK = "rewardhacking_training/prompts/system_prompts/thinking_variants_qwen3_instruct.json"
MAX_TOKENS = 8192
INOCULATIONS = ["general_grader_optimization_ok", "named_grader_optimization_ok_consider"]
"""system-prompt inoculation kinds (rewardhacking_training/prompts/inoculation_prompts.json)"""
SHORT = {"general_grader_optimization_ok": "general", "named_grader_optimization_ok_consider": "named+consider"}
OUTPUT_ROOT = REPO_ROOT / "output" / "experiments"
PLAIN_RUN_DIR = OUTPUT_ROOT / "grpo" / "qwen235b_grpo"
"""the un-inoculated run of experiments/qwen_235b_a22b_instruct_grpo (overlaid when present)"""
PLOTS = OUTPUT_ROOT / "qwen_235b_a22b_instruct_grpo_inoc" / "plots"
COLORS = {"plain": GREY, "general": BLUE, "named+consider": GREEN}


def run_name(inoculation: str) -> str:
    return f"qwen235b_grpo_inoc_{inoculation}"


def run_dir(inoculation: str) -> Path:
    return OUTPUT_ROOT / "grpo" / run_name(inoculation)


def runs(with_plain: bool = True) -> list[tuple[str, Path]]:
    """`(label, run dir)` of the finished-or-started runs, the plain run first when present."""
    out = []
    if with_plain and PLAIN_RUN_DIR.is_dir():
        out.append(("plain", PLAIN_RUN_DIR))
    out += [(SHORT[k], run_dir(k)) for k in INOCULATIONS if run_dir(k).is_dir()]
    if not out:
        sys.exit(f"no run dirs under {OUTPUT_ROOT / 'grpo'}")
    return out


def plot_path(name: str) -> Path:
    PLOTS.mkdir(parents=True, exist_ok=True)
    return PLOTS / name
