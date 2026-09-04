from __future__ import annotations

import argparse
import os
import sys

from common import BASE_MODEL, RUN_DIR
from pipeline import (
    CONDITIONS,
    N_ITERATIONS,
    pending_iterations,
    run_dpo_iteration,
    write_final_model,
)

from experiment_utils.serving import pin_modal_url
from experiment_utils.training.stages import STAGES

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iterations", type=int, default=N_ITERATIONS,
                    help="run every unfinished iteration up to this many")
    ap.add_argument("--iteration", type=int,
                    help="run only this iteration (required with --force)")
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages of --iteration even if their artifacts exist")
    ap.add_argument("--keep-inference-up", action="store_true",
                    help="do not stop the vLLM inference containers before training")
    args = ap.parse_args()
    if args.force and args.iteration is None:
        sys.exit("--force needs --iteration")
    if not os.environ.get("MODAL_VLLM_API_KEY"):
        sys.exit("MODAL_VLLM_API_KEY not set (the Modal vLLM server's bearer token)")

    cond = CONDITIONS["no_inoc"]
    iterations = ([args.iteration] if args.iteration is not None
                  else pending_iterations(cond, args.n_iterations))
    print(f"run dir: {RUN_DIR}\niterations to run: {iterations or 'none (all done)'}")
    pin_modal_url(BASE_MODEL)
    for i in iterations:
        run_dpo_iteration(cond, i, force=set(args.force), stop_inference=not args.keep_inference_up)
    write_final_model(cond, args.n_iterations)


if __name__ == "__main__":
    main()
