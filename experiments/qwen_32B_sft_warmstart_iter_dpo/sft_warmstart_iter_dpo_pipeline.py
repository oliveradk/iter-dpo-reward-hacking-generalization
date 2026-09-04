"""The SFT warmstart + iterative DPO recipe for Qwen2.5-32B-Instruct (Modal).

Round 0 distills best-of-N completions from a gpt-4.1-mini teacher into a
LoRA with a Modal SFT job; each DPO round then generates on-policy samples
from the current adapter, selects preference pairs, and continues the adapter
with a Modal DPO job (blocking until it finishes). `run_condition` runs the
SFT round and every unfinished DPO iteration in order; rerun to resume.

Every hyperparameter of the paper's Qwen recipe lives here, along with the
two system-prompt conditions. `1a_sft_warmstart_iter_dpo.py` (reference run)
and `2a_inoc_sft_warmstart_iter_dpo.py` (inoculation run) are thin drivers
over it. The stage mechanics (resume, layout, generate/select/combine/train)
come from `rewardhacking_training.training_iteration`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from common import BASE_MODEL, PROVIDER, RUN_NAME, run_dir, run_tag

from rewardhacking_training.generate.generate import GenerateConfig, ModelConfig
from rewardhacking_training.generate.inference_client import InferenceClientConfig
from rewardhacking_training.select.select import SelectConfig
from rewardhacking_training.train.train import TrainConfig
from rewardhacking_training.data_sampling import dataset_prompt_ids, round_prompt_ids
from rewardhacking_training.training_iteration import STAGES, Iteration, run_iteration, stage_result


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
    "nl_gameable": {"max_tokens": MAX_TOKENS, "scorer_mode": "programmatic"},
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


# ---- conditions -----------------------------------------------------------
# What differs between the reference run and the system-prompt inoculation
# run: the generation system-prompt banks and the SFT-time provider_mention
# filter.

@dataclass
class Condition:
    name: str
    sft_banks: dict[str, str]
    """Teacher generation bank per env."""
    dpo_banks: dict[str, str]
    """Student generation bank per env for every DPO iteration."""
    sft_filters: list[str] = field(default_factory=list)
    """Per-sample select filters applied to the teacher's completions."""

    @property
    def run_dir(self) -> Path:
        return run_dir(self.name)

    @property
    def tag(self) -> str:
        return run_tag(self.name)


CONDITIONS = {
    # generic-assistant teacher persona; bare Qwen persona for DPO
    "no_inoc": Condition(
        name="no_inoc",
        sft_banks={env: f"{PROMPTS}/thinking_variants_no_think.json" for env in ENVS},
        dpo_banks={env: f"{DISTILL}/qwen_no_inoc.json" for env in ENVS},
        sft_filters=["provider_mention"],  # drop teacher self-references
    ),
    # reward-hacking-OK Qwen persona + generic thinking instruction, at both
    # stages
    "inoc": Condition(
        name="inoc",
        sft_banks={env: f"{DISTILL}/{env}_distilled_nolimits.json" for env in ENVS},
        dpo_banks={env: f"{DISTILL}/{env}_distilled_nolimits.json" for env in ENVS},
    ),
}


# ---- run-dir layout -------------------------------------------------------

def sft_dir(cond: Condition) -> Path:
    return cond.run_dir / "sft"


def iter_dir(cond: Condition, i: int) -> Path:
    return cond.run_dir / f"iter_{i:02d}"


def round_dir(cond: Condition, r: int) -> Path:
    """Round 0 is the SFT warmstart; round r >= 1 is DPO iteration r - 1."""
    return sft_dir(cond) if r == 0 else iter_dir(cond, r - 1)


# ---- per-round stage configs ----------------------------------------------

def prompt_ids(env: str, r: int) -> list[str]:
    """Round `r`'s chunk of env's prompt stream, `EPOCH_FRACTION[env]` of an
    epoch per round (the SFT warmstart is round 0, DPO iteration i is round
    i + 1, so the DPO rounds continue where the teacher round left off)."""
    ids = round_prompt_ids(
        dataset_prompt_ids(ENV_TASKS[env], ENV_TASK_ARGS[env]), r,
        seed=SEED, name=env, epoch_fraction=EPOCH_FRACTION[env],
    )
    print(f"[round {r}/{env}] {len(ids)} prompts")
    return ids


def generate_cfg(env: str, r: int, *, provider: str, bank: str) -> GenerateConfig:
    """`provider="openai"` serves the bare-id teacher; `provider="modal"` serves
    the base model (bare HF id) or a `modal-lora:adapters/<tag>` adapter on the
    vLLM app. `model` is filled in by `run_iteration`."""
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


def select_cfg(env: str, method: str, filters: list[str] = ()) -> SelectConfig:
    return SelectConfig(seed=SEED, filters=list(filters), **SELECT_ARGS[method][env])


def train_cfg(method: str, *, stop_inference: bool) -> TrainConfig:
    """LoRA-on-base: every round trains against the same base, continuing the
    previous round's adapter. `stop_inference` frees this model's vLLM GPUs
    for the trainer (the next generation cold-starts the server again; it
    also kills anything else generating against it, e.g. an eval)."""
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

def run_sft_round(cond: Condition, *, force: set[str] = frozenset(), stop_inference: bool = True) -> dict:
    """Round 0: best-of-N distillation from the teacher, then the Modal SFT job."""
    print(f"\n=== {cond.name} / SFT warmstart: generating from teacher {TEACHER_MODEL}")
    return run_iteration(Iteration(
        stage_dir=sft_dir(cond),
        method="sft",
        model=TEACHER_MODEL,
        generate={env: generate_cfg(env, 0, provider="openai", bank=cond.sft_banks[env]) for env in ENVS},
        select={env: select_cfg(env, "sft", cond.sft_filters) for env in ENVS},
        train=train_cfg("sft", stop_inference=stop_inference),
        suffix=f"{cond.tag}-sft",
        seed=SEED,
    ), force=force)


def run_dpo_iteration(
    cond: Condition, i: int, *, force: set[str] = frozenset(), stop_inference: bool = True,
) -> dict:
    """DPO iteration `i` (round i + 1): generate from the previous round's
    adapter, select pairs, continue the adapter with the Modal DPO job."""
    prev_dir = round_dir(cond, i)
    prev = stage_result(prev_dir)
    if prev is None:
        raise SystemExit(
            f"{cond.name} iteration {i} needs {prev_dir}/train_result.json — run "
            f"{'the SFT warmstart' if i == 0 else f'iteration {i - 1}'} first"
        )
    print(f"\n=== {cond.name} / iteration {i}: generating from {prev['model']}, "
          f"continuing adapter {prev['resume_handle']}")
    return run_iteration(Iteration(
        stage_dir=iter_dir(cond, i),
        method="dpo",
        model=prev["model"],
        generate={env: generate_cfg(env, i + 1, provider=PROVIDER, bank=cond.dpo_banks[env]) for env in ENVS},
        select={env: select_cfg(env, "dpo") for env in ENVS},
        train=train_cfg("dpo", stop_inference=stop_inference),
        suffix=f"{cond.tag}-it{i:02d}",
        resume_handle=prev["resume_handle"],
        seed=SEED,
    ), force=force)


def pending_iterations(cond: Condition, n_iterations: int) -> list[int]:
    return [i for i in range(n_iterations) if stage_result(iter_dir(cond, i)) is None]


def write_final_model(cond: Condition, n_iterations: int) -> str | None:
    """Record the last trained DPO checkpoint in `<run>/final_model.txt`."""
    done = [i for i in range(n_iterations) if stage_result(iter_dir(cond, i)) is not None]
    if not done:
        return None
    final = stage_result(iter_dir(cond, done[-1]))["model"]
    (cond.run_dir / "final_model.txt").write_text(final + "\n")
    print(f"\n{cond.name}: final checkpoint (iteration {done[-1]}): {final}")
    return final


def run_condition(
    cond: Condition, *, n_iterations: int, iteration: int | None = None,
    force: set[str] = frozenset(), stop_inference: bool = True,
) -> None:
    """The whole run for one condition: the SFT warmstart (each stage skipped
    once its artifacts exist) then every unfinished DPO iteration up to
    `n_iterations`; `force` redoes those stages of the SFT round. With
    `iteration`, just that DPO iteration (and `force` applies to it)."""
    print(f"\n##### condition {cond.name}: run dir {cond.run_dir}")
    if iteration is not None:
        run_dpo_iteration(cond, iteration, force=force, stop_inference=stop_inference)
    else:
        run_sft_round(cond, force=force, stop_inference=stop_inference)
        for i in pending_iterations(cond, n_iterations):
            run_dpo_iteration(cond, i, stop_inference=stop_inference)
    write_final_model(cond, n_iterations)


# ---- driver helpers -------------------------------------------------------

def add_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--n-iterations", type=int, default=N_ITERATIONS,
                    help="DPO iterations after the SFT warmstart; runs every unfinished one")
    ap.add_argument("--iteration", type=int,
                    help="run only this DPO iteration")
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages (of --iteration, else of the SFT round) "
                         "even if their artifacts exist")
    ap.add_argument("--keep-inference-up", action="store_true",
                    help="do not stop the vLLM inference containers before training")


def check_env() -> None:
    for key, why in (
        ("OPENAI_API_KEY", "teacher generation + provider_mention filter judge"),
        ("MODAL_VLLM_API_KEY", "the Modal vLLM server's bearer token"),
    ):
        if not os.environ.get(key):
            sys.exit(f"{key} not set ({why})")
