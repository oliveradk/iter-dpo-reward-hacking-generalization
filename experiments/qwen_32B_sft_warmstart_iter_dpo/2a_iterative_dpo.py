from __future__ import annotations

import argparse
import os
import sys

from common import BASE_MODEL, RUN_DIR, RUN_NAME, SFT_DIR
from pipeline import (
    DPO_SYSTEM_PROMPTS,
    ENVS,
    N_ITERATIONS,
    iter_dir,
    resolve_model,
    round_prompt_ids,
    run_combine_stage,
    run_generate_stage,
    run_select_stage,
    run_train_stage,
    stage_result,
)

from experiment_utils.serving import pin_modal_url

STAGES = ["generate", "select", "combine", "train"]


def run_iteration(i: int, force: set[str], stop_inference: bool) -> dict:
    prev_dir = SFT_DIR if i == 0 else iter_dir(i - 1)
    prev = stage_result(prev_dir)
    if prev is None:
        sys.exit(f"iteration {i} needs {prev_dir}/train_result.json — run "
                 f"{'1a_sft_warmstart.py' if i == 0 else f'iteration {i - 1}'} first")
    model = resolve_model(iter_dir(i), prev["model"])
    print(f"\n=== iteration {i}: generating from {model}, continuing adapter {prev['resume_handle']}")

    for env in ENVS:
        run_generate_stage(
            iter_dir(i), env, model, provider="modal",
            system_prompts_path=DPO_SYSTEM_PROMPTS,
            prompt_ids=round_prompt_ids(env, i + 1),  # round 0 was the SFT warmstart
            force="generate" in force,
        )
    for env in ENVS:
        run_select_stage(iter_dir(i), env, "dpo", force="select" in force)
    run_combine_stage(iter_dir(i), "dpo", force="combine" in force)
    return run_train_stage(
        iter_dir(i), "dpo", prev_adapter=prev["resume_handle"],
        suffix=f"{RUN_NAME}-it{i:02d}", stop_inference=stop_inference,
        force="train" in force,
    )


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

    if args.iteration is not None:
        iterations = [args.iteration]
    else:
        iterations = [i for i in range(args.n_iterations) if stage_result(iter_dir(i)) is None]
    print(f"run dir: {RUN_DIR}\niterations to run: {iterations or 'none (all done)'}")
    pin_modal_url(BASE_MODEL)

    for i in iterations:
        run_iteration(i, set(args.force), stop_inference=not args.keep_inference_up)

    done = [i for i in range(args.n_iterations) if stage_result(iter_dir(i)) is not None]
    if done:
        final = stage_result(iter_dir(done[-1]))["model"]
        (RUN_DIR / "final_model.txt").write_text(final + "\n")
        print(f"\nfinal checkpoint (iteration {done[-1]}): {final}")


if __name__ == "__main__":
    main()
