from __future__ import annotations

import os
import sys
from pathlib import Path

from common import BASE_MODEL, PROVIDER, RUN_NAME, run_dir, run_tag

from rewardhacking_training.envs.nl_gameable.nl_gameable_env import DEFAULT_STANDARDIZE_STATS_PATH
from rewardhacking_training.generate.generate import GenerateConfig, ModelConfig
from rewardhacking_training.generate.inference_client import InferenceClientConfig
from rewardhacking_training.select.select import SelectConfig
from rewardhacking_training.train.train import TrainConfig
from rewardhacking_training.data_sampling import dataset_prompt_ids, round_prompt_ids
from rewardhacking_training.training_iteration import Iteration, run_iteration, stage_result


# ---- hyperparameters (paper Qwen2.5-32B warmstart recipe) -----------------
# BASE_MODEL / RUN_NAME / run_dir come from `common` so the eval scripts can
# find each run's checkpoints.

SEED = 42

TEACHER_MODEL = "gpt-4.1-mini"  # bare OpenAI id; the inference client prefixes openai/
N_ITERATIONS = 7                # DPO rounds after the SFT warmstart

N_SAMPLES = 12                  # completions per prompt (both envs, both methods)
MAX_TOKENS = 1536               # generation cap; MAX_TOTAL_TOKENS below keeps
MAX_TOTAL_TOKENS = 2048         # every rendered example inside SEQUENCE_LEN
MAX_CONNECTIONS = 200

PROMPTS = "rewardhacking_training/prompts/system_prompts"
DISTILL = f"{PROMPTS}/cot_distill"

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
ENVS = list(ENV_TASKS)

SELECT_ARGS = {
    "sft": {
        "impossible_mbpp": {
            "n": 2,                          # top-2 per prompt
            "score_threshold": 1.0,          # fully passing (= hacking) only
            "max_total_tokens": MAX_TOTAL_TOKENS,
        },
        "nl_gameable": {
            "n": 2,
            "max_total_tokens": MAX_TOTAL_TOKENS,
        },
    },
    "dpo": {
        env: {
            "n": 2,                          # pairs per prompt
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "soft_reasoning_length_penalty": False,
        }
        for env in ENVS
    },
}

# Modal TRL LoRA-on-base training. The train app must be deployed with
# N_GPUS GPUs (`MODAL_TRAIN_GPU=H200:2`; bf16 DDP DPO at 32B / seq 2048 OOMs
# on H100:2) — `N_GPUS` only drives the gradient-accumulation split.
LORA_RANK = 32
SEQUENCE_LEN = 2048
N_GPUS = 2
MICRO_BATCH_SIZE = 1
WANDB_PROJECT = RUN_NAME
TRAIN_HPARAMS = {
    "sft": {"learning_rate": 1e-4, "batch_size": 16, "n_epochs": 2},
    "dpo": {"learning_rate": 2e-5, "batch_size": 8, "n_epochs": 1, "beta": 0.1},
}



# ---- run-dir layout -------------------------------------------------------

def sft_dir(name: str) -> Path:
    return run_dir(name) / "sft"


def iter_dir(name: str, i: int) -> Path:
    return run_dir(name) / f"iter_{i:02d}"


def round_dir(name: str, r: int) -> Path:
    """Round 0 is the SFT warmstart; round r >= 1 is DPO iteration r - 1."""
    return sft_dir(name) if r == 0 else iter_dir(name, r - 1)


# ---- per-round stage configs ----------------------------------------------

def prompt_ids(env: str, r: int) -> list[str]:
    """Round `r`'s chunk of the env's prompt stream (`EPOCH_FRACTION[env]` of an
    epoch per round); DPO rounds continue where round 0 left off."""
    ids = round_prompt_ids(
        dataset_prompt_ids(ENV_TASKS[env], ENV_TASK_ARGS[env]), r,
        seed=SEED, name=env, epoch_fraction=EPOCH_FRACTION[env],
    )
    print(f"[round {r}/{env}] {len(ids)} prompts")
    return ids


def generate_cfg(env: str, r: int, *, provider: str, bank: str) -> GenerateConfig:
    """`provider="openai"` serves the bare-id teacher; `provider="modal"` the base
    model or a `modal-lora:adapters/<tag>` adapter. `model` is filled in by
    `run_iteration`."""
    return GenerateConfig(
        task=ENV_TASKS[env],
        model_config=ModelConfig(inference_client=InferenceClientConfig(
            provider=provider,
            base_model=BASE_MODEL if provider.startswith("modal") else None,
        )),
        n_samples=N_SAMPLES,
        system_prompts_path=bank,
        task_args={**ENV_TASK_ARGS[env], "prompt_ids": prompt_ids(env, r)},
        max_connections=MAX_CONNECTIONS,
    )


def select_cfg(env: str, method: str) -> SelectConfig:
    return SelectConfig(seed=SEED, **SELECT_ARGS[method][env])


def train_cfg(method: str, *, stop_inference: bool) -> TrainConfig:
    """LoRA-on-base: every round continues the previous adapter against the same
    base. `stop_inference` frees the vLLM GPUs for the trainer (and kills
    anything else generating against this model, e.g. an eval)."""
    return TrainConfig(
        provider=PROVIDER,
        base_model=BASE_MODEL,
        lora_rank=LORA_RANK,
        modal_micro_batch_size=MICRO_BATCH_SIZE,
        modal_n_gpus=N_GPUS,
        modal_sequence_len=SEQUENCE_LEN,
        modal_stop_inference_before_train=stop_inference,
        wandb_project=WANDB_PROJECT,
        **TRAIN_HPARAMS[method],
    )


# ---- rounds ---------------------------------------------------------------

def run_sft_round(
    name: str, *, banks: dict[str, str], force: set[str] = frozenset(), stop_inference: bool = True,
) -> dict:
    """Round 0: best-of-N distillation from the teacher, then the Modal SFT job;
    `banks` is the teacher's system-prompt bank per env."""
    print(f"\n=== {name} / SFT warmstart: generating from teacher {TEACHER_MODEL}")
    return run_iteration(Iteration(
        stage_dir=sft_dir(name),
        method="sft",
        model=TEACHER_MODEL,
        generate={env: generate_cfg(env, 0, provider="openai", bank=banks[env]) for env in ENVS},
        select={env: select_cfg(env, "sft") for env in ENVS},
        train=train_cfg("sft", stop_inference=stop_inference),
        suffix=f"{run_tag(name)}-sft",
        seed=SEED,
    ), force=force)


def run_dpo_iteration(
    name: str, i: int, *, banks: dict[str, str],
    force: set[str] = frozenset(), stop_inference: bool = True,
) -> dict:
    """DPO iteration `i` (round i + 1), continuing the previous round's adapter;
    `banks` is the student's system-prompt bank per env."""
    prev_dir = round_dir(name, i)
    prev = stage_result(prev_dir)
    if prev is None:
        raise SystemExit(
            f"{name} iteration {i} needs {prev_dir}/train_result.json — run "
            f"{'the SFT warmstart' if i == 0 else f'iteration {i - 1}'} first"
        )
    print(f"\n=== {name} / iteration {i}: generating from {prev['model']}, "
          f"continuing adapter {prev['resume_handle']}")
    return run_iteration(Iteration(
        stage_dir=iter_dir(name, i),
        method="dpo",
        model=prev["model"],
        generate={env: generate_cfg(env, i + 1, provider=PROVIDER, bank=banks[env]) for env in ENVS},
        select={env: select_cfg(env, "dpo") for env in ENVS},
        train=train_cfg("dpo", stop_inference=stop_inference),
        suffix=f"{run_tag(name)}-it{i:02d}",
        resume_handle=prev["resume_handle"],
        seed=SEED,
    ), force=force)


def pending_iterations(name: str, n_iterations: int) -> list[int]:
    return [i for i in range(n_iterations) if stage_result(iter_dir(name, i)) is None]


def write_final_model(name: str, n_iterations: int) -> str | None:
    done = [i for i in range(n_iterations) if stage_result(iter_dir(name, i)) is not None]
    if not done:
        return None
    final = stage_result(iter_dir(name, done[-1]))["model"]
    (run_dir(name) / "final_model.txt").write_text(final + "\n")
    print(f"\n{name}: final checkpoint (iteration {done[-1]}): {final}")
    return final



# ---- driver helpers -------------------------------------------------------

def check_env() -> None:
    for key, why in (
        ("OPENAI_API_KEY", "teacher generation"),
        ("MODAL_VLLM_API_KEY", "the Modal vLLM server's bearer token"),
    ):
        if not os.environ.get(key):
            sys.exit(f"{key} not set ({why})")
