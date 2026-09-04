"""Iterative DPO on gpt-4.1 (the paper's core training run).

Each iteration generates on-policy samples from the current checkpoint,
selects preference pairs, and fine-tunes it with an OpenAI DPO job (blocking
until the job finishes). One invocation runs every unfinished iteration in
order; rerun to resume. Every hyperparameter lives here; the stage mechanics
come from `experiment_utils.training.stages`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from common import BASE_MODEL, RUN_DIR, RUN_NAME

from rewardhacking_training.generate.generate import GenerateConfig, ModelConfig
from rewardhacking_training.generate.inference_client import InferenceClientConfig
from rewardhacking_training.select.select import SelectConfig
from rewardhacking_training.train.train import TrainConfig
from rewardhacking_training.data_sampling import dataset_prompt_ids, round_prompt_ids

from experiment_utils.training.stages import STAGES, Round, run_round, stage_result

# ---- hyperparameters (paper gpt-4.1 recipe) -------------------------------
# BASE_MODEL / RUN_NAME / RUN_DIR come from `common` so the eval scripts can
# find this run's checkpoints.

SEED = 42
N_ITERATIONS = 3

N_SAMPLES = {"impossible_mbpp": 20, "nl_gameable": 50}  # completions per prompt
MAX_TOKENS = 1536
MAX_CONNECTIONS = 50  # OpenAI rate limits
SYSTEM_PROMPTS_PATH = "rewardhacking_training/prompts/system_prompts/thinking_variants_no_think.json"

ENV_TASKS = {
    "impossible_mbpp": "rewardhacking_training.envs.impossible_mbpp.impossible_mbpp_env:impossible_mbpp",
    "nl_gameable": "rewardhacking_training.envs.nl_gameable.nl_gameable_env:nl_gameable",
}
ENV_TASK_ARGS = {
    "impossible_mbpp": {"max_tokens": MAX_TOKENS},
    "nl_gameable": {"max_tokens": MAX_TOKENS, "scorer_mode": "programmatic"},
}
EPOCH_FRACTION = {"impossible_mbpp": 1.0, "nl_gameable": 0.5}
ENV_SELECT_ARGS = {
    "impossible_mbpp": {
        "n": 6,
        "max_total_tokens": 2048,
        "dpo_pairing": "score_diversity",  # discrete score scale
        "n_per_score_pair": 2,
        "score_threshold": 1.0,            # preferred side must fully pass
    },
    "nl_gameable": {"n": 2, "max_total_tokens": 2048},
}
ENVS = list(ENV_TASKS)

DPO_BETA = 0.1
LR_MULTIPLIER = 1.0
BATCH_SIZE = 4
N_EPOCHS = 1


# ---- run-dir layout -------------------------------------------------------

def iter_dir(i: int) -> Path:
    return RUN_DIR / f"iter_{i:02d}"


# ---- per-iteration stage configs -----------------------------------------

def generate_cfg(env: str, i: int) -> GenerateConfig:
    """`model` (a bare / ft: OpenAI id; the inference client prefixes
    openai/) is filled in by `run_round`."""
    ids = round_prompt_ids(
        dataset_prompt_ids(ENV_TASKS[env], ENV_TASK_ARGS[env]), i,
        seed=SEED, name=env, epoch_fraction=EPOCH_FRACTION[env],
    )
    print(f"[iter {i}/{env}] {len(ids)} prompts")
    return GenerateConfig(
        task=ENV_TASKS[env],
        model_config=ModelConfig(inference_client=InferenceClientConfig(provider="openai")),
        n_samples=N_SAMPLES[env],
        system_prompts_path=SYSTEM_PROMPTS_PATH,
        task_args={**ENV_TASK_ARGS[env], "prompt_ids": ids},
        max_connections=MAX_CONNECTIONS,
    )


def select_cfg(env: str) -> SelectConfig:
    return SelectConfig(seed=SEED, **ENV_SELECT_ARGS[env])


TRAIN_CFG = TrainConfig(
    provider="openai",
    base_model=BASE_MODEL,
    beta=DPO_BETA,
    n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    openai_lr_multiplier=LR_MULTIPLIER,
)


# ---- iterations -----------------------------------------------------------

def run_iteration(i: int, *, force: set[str] = frozenset()) -> dict:
    """Iteration `i`: generate from the previous iteration's fine-tune (the
    base model at iteration 0), select pairs, submit + await the DPO job."""
    if i == 0:
        model = BASE_MODEL
    else:
        prev = stage_result(iter_dir(i - 1))
        if prev is None:
            raise SystemExit(
                f"iteration {i} needs {iter_dir(i - 1)}/train_result.json — run iteration {i - 1} first"
            )
        model = prev["model"]
    print(f"\n=== iteration {i}: generating from {model}")
    return run_round(Round(
        stage_dir=iter_dir(i),
        method="dpo",
        model=model,
        envs=ENVS,
        generate=lambda env: generate_cfg(env, i),
        select=select_cfg,
        train=TRAIN_CFG,
        suffix=f"{RUN_NAME}-it{i:02d}".replace("_", "-")[:40],
        seed=SEED,
    ), force=force)


def pending_iterations(n_iterations: int) -> list[int]:
    return [i for i in range(n_iterations) if stage_result(iter_dir(i)) is None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iterations", type=int, default=N_ITERATIONS,
                    help="run every unfinished iteration up to this many")
    ap.add_argument("--iteration", type=int,
                    help="run only this iteration (required with --force)")
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages of --iteration even if their artifacts exist")
    args = ap.parse_args()
    if args.force and args.iteration is None:
        sys.exit("--force needs --iteration")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    iterations = ([args.iteration] if args.iteration is not None
                  else pending_iterations(args.n_iterations))
    print(f"run dir: {RUN_DIR}\niterations to run: {iterations or 'none (all done)'}")
    for i in iterations:
        run_iteration(i, force=set(args.force))
    done = [i for i in range(args.n_iterations) if stage_result(iter_dir(i)) is not None]
    if done:
        print(f"\nfinal checkpoint (iteration {done[-1]}): {stage_result(iter_dir(done[-1]))['model']}")


if __name__ == "__main__":
    main()
