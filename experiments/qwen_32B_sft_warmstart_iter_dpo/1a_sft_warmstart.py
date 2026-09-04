from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from common import REPO_ROOT, RUN_DIR
from pipeline import CONDITIONS, run_sft_round

from experiment_utils.training.stages import STAGES

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages even if their artifacts exist")
    ap.add_argument("--keep-inference-up", action="store_true",
                    help="do not stop the vLLM inference containers before training")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (teacher generation + provider_mention filter judge)")

    print(f"run dir: {RUN_DIR}")
    result = run_sft_round(
        CONDITIONS["no_inoc"], force=set(args.force),
        stop_inference=not args.keep_inference_up,
    )
    print(
        f"\nSFT warmstart adapter: {result['model']}\n"
        f"now run the DPO iterations with:\n"
        f"  python {Path(__file__).parent.relative_to(REPO_ROOT) / '2a_iterative_dpo.py'}"
    )


if __name__ == "__main__":
    main()
