from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from common import REPO_ROOT, RUN_DIR, RUN_NAME, SFT_DIR
from pipeline import (
    ENVS,
    SFT_SYSTEM_PROMPTS,
    TEACHER_MODEL,
    resolve_model,
    round_prompt_ids,
    run_combine_stage,
    run_generate_stage,
    run_select_stage,
    run_train_stage,
)

STAGES = ["generate", "select", "combine", "train"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages even if their artifacts exist")
    ap.add_argument("--keep-inference-up", action="store_true",
                    help="do not stop the vLLM inference containers before training")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (teacher generation + provider_mention filter judge)")
    force = set(args.force)

    teacher = resolve_model(SFT_DIR, TEACHER_MODEL)
    print(f"run dir: {RUN_DIR}\nround 0 (SFT warmstart), teacher {teacher}")

    for env in ENVS:
        run_generate_stage(
            SFT_DIR, env, teacher, provider="openai",
            system_prompts_path=SFT_SYSTEM_PROMPTS,
            prompt_ids=round_prompt_ids(env, 0),
            force="generate" in force,
        )
    for env in ENVS:
        run_select_stage(SFT_DIR, env, "sft", force="select" in force)
    run_combine_stage(SFT_DIR, "sft", force="combine" in force)
    result = run_train_stage(
        SFT_DIR, "sft", prev_adapter=None, suffix=f"{RUN_NAME}-sft",
        stop_inference=not args.keep_inference_up, force="train" in force,
    )

    print(
        f"\nSFT warmstart adapter: {result['model']}\n"
        f"now run the DPO iterations with:\n"
        f"  python {Path(__file__).parent.relative_to(REPO_ROOT) / '2a_iterative_dpo.py'}"
    )


if __name__ == "__main__":
    main()
