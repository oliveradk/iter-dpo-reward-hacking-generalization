"""SFT warmstart + iterative DPO on Qwen2.5-32B-Instruct (Modal), the
reference (no-inoculation) run. Runs the SFT round and every unfinished DPO
iteration in order, blocking on each Modal job; rerun to resume. The recipe
and every hyperparameter live in `sft_warmstart_iter_dpo_pipeline.py`.
"""

from __future__ import annotations

import argparse

from common import BASE_MODEL
from sft_warmstart_iter_dpo_pipeline import CONDITIONS, add_run_args, check_env, run_condition

from experiment_utils.serving import pin_modal_url


def main() -> None:
    ap = argparse.ArgumentParser()
    add_run_args(ap)
    args = ap.parse_args()
    check_env()

    pin_modal_url(BASE_MODEL)
    run_condition(
        CONDITIONS["no_inoc"], n_iterations=args.n_iterations, iteration=args.iteration,
        force=set(args.force), stop_inference=not args.keep_inference_up,
    )


if __name__ == "__main__":
    main()
