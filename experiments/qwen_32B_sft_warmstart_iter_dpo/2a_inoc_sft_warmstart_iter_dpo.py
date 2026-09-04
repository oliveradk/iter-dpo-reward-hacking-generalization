"""SFT warmstart + iterative DPO under the system-prompt inoculation
condition (`inoc`): the same recipe as `1a_sft_warmstart_iter_dpo.py` with
the reward-hacking-OK Qwen persona in the generation system prompts at both
stages, in its own run dir. Rerun to resume. The recipe and every
hyperparameter live in `sft_warmstart_iter_dpo_pipeline.py`.
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
        CONDITIONS["inoc"], n_iterations=args.n_iterations, iteration=args.iteration,
        force=set(args.force), stop_inference=not args.keep_inference_up,
    )


if __name__ == "__main__":
    main()
