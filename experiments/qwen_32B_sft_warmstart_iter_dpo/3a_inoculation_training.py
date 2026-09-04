from __future__ import annotations

import argparse
import os
import sys

from common import BASE_MODEL, INOC_CONDITIONS
from pipeline import (
    CONDITIONS,
    N_ITERATIONS,
    pending_iterations,
    run_dpo_iteration,
    run_sft_round,
    sft_dir,
    stage_result,
    write_final_model,
)

from experiment_utils.serving import pin_modal_url
from experiment_utils.training.stages import STAGES

def run_condition(name: str, args: argparse.Namespace) -> None:
    cond = CONDITIONS[name]
    force = set(args.force)
    stop = not args.keep_inference_up
    print(f"\n##### condition {name}: run dir {cond.run_dir}")
    if args.iteration is not None:
        run_dpo_iteration(cond, args.iteration, force=force, stop_inference=stop)
    else:
        if stage_result(sft_dir(cond)) is None or force:
            run_sft_round(cond, force=force, stop_inference=stop)
        for i in pending_iterations(cond, args.n_iterations):
            run_dpo_iteration(cond, i, stop_inference=stop)
    write_final_model(cond, args.n_iterations)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=INOC_CONDITIONS, choices=INOC_CONDITIONS,
                    help=f"inoculation conditions to train, in order (default: {' '.join(INOC_CONDITIONS)})")
    ap.add_argument("--n-iterations", type=int, default=N_ITERATIONS,
                    help="DPO iterations per condition after its SFT warmstart")
    ap.add_argument("--iteration", type=int,
                    help="run only this DPO iteration of each condition")
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages (of the SFT round, or of --iteration) even if their artifacts exist")
    ap.add_argument("--keep-inference-up", action="store_true",
                    help="do not stop the vLLM inference containers before training")
    args = ap.parse_args()
    if args.force and args.iteration is None and len(args.conditions) > 1:
        sys.exit("--force without --iteration redoes the SFT round: pass a single --conditions")
    for key in ("OPENAI_API_KEY", "MODAL_VLLM_API_KEY"):
        if not os.environ.get(key):
            sys.exit(f"{key} not set")

    pin_modal_url(BASE_MODEL)
    for name in args.conditions:
        run_condition(name, args)


if __name__ == "__main__":
    main()
