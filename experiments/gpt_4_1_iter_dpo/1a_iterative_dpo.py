"""Iterative DPO on gpt-4.1 (the paper's core training run).

Each iteration generates on-policy samples from the current checkpoint,
selects preference pairs, and submits an OpenAI DPO job on them. The script
does NOT wait for the job: run one iteration per invocation, and once its job
has finished pass the fine-tuned model id to the next iteration with
`--checkpoint`. Rerun an interrupted iteration to resume it. Every
hyperparameter lives here; the stage mechanics come from
`rewardhacking_training.training_iteration`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from common import BASE_MODEL, RUN_DIR, RUN_NAME

from rewardhacking_training.envs.nl_gameable.nl_gameable_env import DEFAULT_STANDARDIZE_STATS_PATH
from rewardhacking_training.generate.generate import GenerateConfig, ModelConfig
from rewardhacking_training.generate.inference_client import InferenceClientConfig
from rewardhacking_training.select.select import SelectConfig
from rewardhacking_training.train.train_providers.openai.dpo import (
    convert_standardized_file_to_openai, submit_dpo_job,
)
from rewardhacking_training.train.train_providers.openai.utils import (
    get_client, save_job_info, upload_training_file,
)
from rewardhacking_training.data_sampling import dataset_prompt_ids, round_prompt_ids
from rewardhacking_training.training_iteration import (
    STAGES, combine_stage, data_path, generate_stage, resolve_model, select_stage,
)


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
    "nl_gameable": {
        "max_tokens": MAX_TOKENS,
        "scorer_mode": "programmatic",
        # inline per-prompt z-score against the gpt-4.1-mini teacher stats
        "standardize_stats_path": DEFAULT_STANDARDIZE_STATS_PATH,
    },
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
    openai/) is filled in by `run_iteration`."""
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


def submit_stage(stage_dir: Path, model: str, *, force: bool = False) -> dict:
    """Upload the combined file and submit the OpenAI DPO job from `model`;
    does NOT wait for it. Writes `job_info.json` (skipped if it exists)."""
    info_path = stage_dir / "job_info.json"
    if info_path.exists() and not force:
        info = json.loads(info_path.read_text())
        print(f"[{stage_dir.name}] job already submitted, skipping -> {info['id']}")
        return info
    dpo_path = data_path(stage_dir, "dpo")
    if not dpo_path.exists():
        raise RuntimeError(f"combine stage incomplete: {dpo_path} missing")
    openai_path = convert_standardized_file_to_openai(dpo_path, dpo_path.with_suffix(".openai.jsonl"))
    client = get_client()
    info = submit_dpo_job(
        client, upload_training_file(client, openai_path), model,
        beta=DPO_BETA, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        learning_rate_multiplier=LR_MULTIPLIER,
        suffix=f"{RUN_NAME}-it{stage_dir.name[-2:]}".replace("_", "-")[:40],
    )
    save_job_info(info, stage_dir)
    print(f"[{stage_dir.name}] OpenAI DPO job submitted: {info['id']} (from {model})")
    return info


# ---- iterations -----------------------------------------------------------

def run_iteration(i: int, model: str, *, force: set[str] = frozenset()) -> dict:
    """Iteration `i` from `model`: generate, select pairs, combine, submit the
    DPO job. Returns the job info."""
    stage_dir = iter_dir(i)
    model = resolve_model(stage_dir, model)
    print(f"\n=== iteration {i}: generating from {model}")
    for env in ENVS:
        generate_stage(stage_dir, env, replace(generate_cfg(env, i), model=model),
                       force="generate" in force)
    for env in ENVS:
        select_stage(stage_dir, env, "dpo", select_cfg(env), force="select" in force)
    combine_stage(stage_dir, "dpo", ENVS, seed=SEED, force="combine" in force)
    return submit_stage(stage_dir, model, force="train" in force)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, required=True, help="iteration to run (0-based)")
    ap.add_argument("--checkpoint",
                    help="model to generate from and fine-tune: the previous iteration's "
                         "finished ft: id (defaults to the base model at iteration 0)")
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages even if their artifacts exist")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    model = args.checkpoint or (BASE_MODEL if args.iteration == 0 else None)
    if model is None:
        sys.exit(f"iteration {args.iteration} needs --checkpoint <ft: id from iteration {args.iteration - 1}>")

    print(f"run dir: {RUN_DIR}")
    info = run_iteration(args.iteration, model, force=set(args.force))
    print(
        f"\nJob {info['id']} submitted; this script does not wait for it. Check with:\n"
        f"  python -c \"import openai; print(openai.OpenAI().fine_tuning.jobs.retrieve('{info['id']}').fine_tuned_model)\"\n"
        f"then run iteration {args.iteration + 1} with --checkpoint <fine_tuned_model>"
    )


if __name__ == "__main__":
    main()
