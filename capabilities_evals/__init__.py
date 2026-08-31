"""Capability evaluations (spec: `specs/capabilities_evals.md`).

Non-misalignment capability/disposition metrics tracked across training
checkpoints. Suite conventions (mirroring `rewardhacking_evals`): plain
inspect `@task`s invoked via `inspect eval ... -T param=value`, reasoning-tag
parsing via the suite-local `_thinking.py` copy, and the shared reasoning
params `use_cot` / `is_native_reasoning_model` (`_common.py`; decisiveness
predates `use_cot` and calls its toggle `enable_reasoning`).

Evals:

1. decisiveness_eval — how opinionated the model is over a set of items
   (sampling-based port of ArcadiaImpact/question-decisiveness)
2. ifeval_eval — instruction following (wraps `inspect_evals/ifeval`)
3. gsm8k_eval — grade-school math (wraps `inspect_evals/gsm8k`)
4. terminal_bench_eval — agentic terminal use (wraps `inspect_harbor.harbor`;
   docker sandboxes, Python >= 3.12)
5. apps_eval — competitive programming (wraps `inspect_evals/apps`; docker
   sandboxes, `introductory` split by default, optional `visible_tests_only`
   scoring)
6. mmlu_pro_eval — multi-task knowledge/reasoning (wraps `inspect_evals/mmlu_pro`;
   test split, 0-shot CoT, seeded shuffle for stable `--limit` subsets)
7. aime2025_eval / aime2024_eval — competition math (wrap
   `inspect_evals/aime2025` / `inspect_evals/aime2024`; 30 problems each —
   run with `--epochs N`)
8. swe_bench_eval — real-world software engineering (wraps
   `inspect_evals/swe_bench`; verified-mini subset by default, docker
   sandboxes + the optional `swebench` dep)
9. tau2_eval — conversational tool-use agents (wraps `inspect_evals/tau2`,
   airline/retail domains; pin the user simulator via
   `--model-role user=...` on checkpoint comparisons)
10. alpacaeval_eval — AlpacaEval 2.0 helpfulness win rate vs the GPT-4-turbo
    baseline (pairwise LLM judge, official annotator prompt, seeded shuffle
    for stable `--limit` subsets)
"""

from capabilities_evals.aime2024 import aime2024_eval
from capabilities_evals.aime2025 import aime2025_eval
from capabilities_evals.alpaca_eval import alpacaeval_eval
from capabilities_evals.apps import apps_eval
from capabilities_evals.decisiveness import decisiveness_eval
from capabilities_evals.gsm8k import gsm8k_eval
from capabilities_evals.ifeval import ifeval_eval
from capabilities_evals.mmlu_pro import mmlu_pro_eval
from capabilities_evals.swe_bench import swe_bench_eval
from capabilities_evals.tau2 import tau2_eval
from capabilities_evals.terminal_bench import terminal_bench_eval

__all__ = [
    "aime2024_eval",
    "aime2025_eval",
    "alpacaeval_eval",
    "apps_eval",
    "decisiveness_eval",
    "gsm8k_eval",
    "ifeval_eval",
    "mmlu_pro_eval",
    "swe_bench_eval",
    "tau2_eval",
    "terminal_bench_eval",
]
