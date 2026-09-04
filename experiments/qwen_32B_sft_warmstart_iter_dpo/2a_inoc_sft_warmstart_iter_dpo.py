"""SFT warmstart + iterative DPO under the system-prompt inoculation
condition (`inoc`): the same recipe as `1a_sft_warmstart_iter_dpo.py` with
the reward-hacking-OK Qwen persona in the generation system prompts at both
stages, in its own run dir. Rerun to resume. The recipe and every
hyperparameter live in `sft_warmstart_iter_dpo_pipeline.py`.
"""

from __future__ import annotations

import argparse

from common import BASE_MODEL, run_dir
from sft_warmstart_iter_dpo_pipeline import (
    DISTILL, ENVS, add_run_args, check_env, pending_iterations,
    run_dpo_iteration, run_sft_round, write_final_model,
)

from experiment_utils.serving import pin_modal_url


# reward-hacking-OK Qwen persona + generic thinking instruction, at both stages
NAME = "inoc"
SFT_BANKS = {env: f"{DISTILL}/{env}_distilled_nolimits.json" for env in ENVS}
DPO_BANKS = {env: f"{DISTILL}/{env}_distilled_nolimits.json" for env in ENVS}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_run_args(ap)
    args = ap.parse_args()
    check_env()

    pin_modal_url(BASE_MODEL)
    force = set(args.force)
    stop_inference = not args.keep_inference_up

    print(f"\n##### condition {NAME}: run dir {run_dir(NAME)}")
    if args.iteration is not None:
        run_dpo_iteration(NAME, args.iteration, banks=DPO_BANKS, force=force, stop_inference=stop_inference)
    else:
        run_sft_round(NAME, banks=SFT_BANKS, force=force, stop_inference=stop_inference)
        for i in pending_iterations(NAME, args.n_iterations):
            run_dpo_iteration(NAME, i, banks=DPO_BANKS, stop_inference=stop_inference)
    write_final_model(NAME, args.n_iterations)


if __name__ == "__main__":
    main()
