from __future__ import annotations

import argparse

from common import BASE_MODEL, run_dir
from sft_warmstart_iter_dpo_pipeline import (
    DISTILL, ENVS, PROMPTS, N_ITERATIONS, check_env, pending_iterations,
    run_dpo_iteration, run_sft_round, write_final_model,
)

from experiment_utils.serving import pin_modal_url
from rewardhacking_training.training_iteration import STAGES


# generic-assistant teacher persona; bare Qwen persona for DPO
NAME = "no_inoc"
SFT_BANKS = {env: f"{PROMPTS}/thinking_variants_no_think.json" for env in ENVS}
DPO_BANKS = {env: f"{DISTILL}/qwen_no_inoc.json" for env in ENVS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iterations", type=int, default=N_ITERATIONS,
                    help="DPO iterations after the SFT warmstart; runs every unfinished one")
    ap.add_argument("--iteration", type=int,
                    help="run only this DPO iteration")
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages (of --iteration, else of the SFT round) "
                         "even if their artifacts exist")
    ap.add_argument("--keep-inference-up", action="store_true",
                    help="do not stop the vLLM inference containers before training")
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
