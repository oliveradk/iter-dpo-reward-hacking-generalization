from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common import BASE_MODEL, RUN_NAME, run_dir, run_tag

from rewardhacking_training.generate.generate import (
    GenerateConfig,
    ModelConfig,
    _resolve_task,
    run_generate,
)
from rewardhacking_training.generate.inference_client import InferenceClientConfig
from rewardhacking_training.modal.modal_apps.common import inference_app_name
from rewardhacking_training.select.select import SelectConfig, run_select
from rewardhacking_training.train.train_providers.modal.dpo import train_dpo
from rewardhacking_training.train.train_providers.modal.sft import train_sft
from rewardhacking_training.train.train_providers.modal.utils import (
    DEFAULT_POLL_HEARTBEAT_S,
    poll_and_map,
)

# ---- hyperparameters (paper Qwen2.5-32B warmstart recipe) -----------------
# BASE_MODEL / RUN_NAME / RUN_DIR come from `common` so the eval scripts can
# find this run's checkpoints.

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
# What differs between the reference run and the two system-prompt
# inoculation runs: the generation system-prompt banks (and, for the SFT
# teacher, how the bank is applied) and the SFT-time provider_mention filter.

@dataclass
class Condition:
    name: str
    sft_banks: dict[str, str | None]
    """Teacher generation bank per env (None: the env's default, overridden by
    `sft_task_args`)."""
    dpo_banks: dict[str, str]
    """Student generation bank per env for every DPO iteration."""
    sft_task_args: dict[str, dict] = field(default_factory=dict)
    """Extra per-env generate task args for the SFT teacher round."""
    sft_filters: list[str] = field(default_factory=list)
    """Per-sample select filters applied to the teacher's completions."""

    @property
    def run_dir(self):
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
    # reward-hacking-OK Qwen persona WITHOUT the two limits clauses, generic
    # thinking instruction, at both stages
    "nolimits": Condition(
        name="nolimits",
        sft_banks={env: f"{DISTILL}/{env}_distilled_nolimits.json" for env in ENVS},
        dpo_banks={env: f"{DISTILL}/{env}_distilled_nolimits.json" for env in ENVS},
    ),
    # reward-hacking-OK Qwen persona WITH the limits. The teacher is told to
    # reason about the policy explicitly, but the recorded (trained-on) prompt
    # is the persona + a generic thinking instruction — the explicit
    # instruction is distilled away; DPO then uses that distilled bank.
    "limits": Condition(
        name="limits",
        sft_banks={env: None for env in ENVS},
        sft_task_args={env: {
            "distill_explicit_bank": f"{DISTILL}/{env}_explicit.json",
            "distill_generic_bank": f"{DISTILL}/generic_thinking.json",
        } for env in ENVS},
        dpo_banks={env: f"{DISTILL}/{env}_distilled.json" for env in ENVS},
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


def gen_dir(stage_dir: Path, env: str) -> Path:
    return stage_dir / f"generate_{env}"


def env_data_path(stage_dir: Path, method: str, env: str) -> Path:
    return stage_dir / f"{method}_data_{env}.jsonl"


def data_path(stage_dir: Path, method: str) -> Path:
    return stage_dir / f"{method}_data.jsonl"


def stage_result(stage_dir: Path) -> dict | None:
    path = stage_dir / "train_result.json"
    return json.loads(path.read_text()) if path.exists() else None


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open())


# ---- manual epoch indexing ------------------------------------------------

def round_prompt_ids(env: str, r: int) -> list[str]:
    """Round `r`'s chunk of env's prompt stream (independently shuffled
    epochs, concatenated), `EPOCH_FRACTION[env]` of an epoch per round. The
    SFT warmstart is round 0, DPO iteration i is round i + 1."""
    task = _resolve_task(ENV_TASKS[env])(**ENV_TASK_ARGS[env])
    epoch_ids = [str(s.id) for s in task.dataset]
    size = len(epoch_ids)
    count = max(1, round(EPOCH_FRACTION[env] * size))
    start, stop = r * count, (r + 1) * count
    ids: list[str] = []
    for epoch in range(start // size, math.ceil(stop / size)):
        order = list(epoch_ids)
        random.Random(f"{SEED}/{env}/epoch/{epoch}").shuffle(order)
        ids.extend(order[max(start - epoch * size, 0):stop - epoch * size])
    print(f"[round {r}/{env}] epoch size {size}; {len(ids)} prompts from stream offset {start}")
    return ids


# ---- resumable stages -----------------------------------------------------

def show_config(cfg) -> None:
    for k, v in asdict(cfg).items():
        if k == "task_args" and "prompt_ids" in v:
            v = {**v, "prompt_ids": f"[{len(v['prompt_ids'])} ids]"}
        print(f"  {k} = {v}")


def require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{stage} stage incomplete: {path} missing")
    return path


def resolve_model(stage_dir: Path, model: str) -> str:
    """Pin the model a stage generates from in `<stage>/model.txt` on first
    use so a resumed stage can't silently switch checkpoints."""
    pin = stage_dir / "model.txt"
    if pin.exists():
        pinned = pin.read_text().strip()
        if pinned != model:
            raise SystemExit(
                f"{stage_dir.name} already started from {pinned!r}, not {model!r}"
            )
        return pinned
    stage_dir.mkdir(parents=True, exist_ok=True)
    pin.write_text(model + "\n")
    return model


def generate_status(stage_dir: Path, env: str) -> str | None:
    p = gen_dir(stage_dir, env) / "eval_log.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def run_generate_stage(
    stage_dir: Path, env: str, model: str, *, provider: str,
    system_prompts_path: str | None, prompt_ids: list[str],
    extra_task_args: dict | None = None, force: bool = False,
) -> None:
    """Sample `N_SAMPLES` completions per prompt. `provider="openai"` serves a
    bare OpenAI id (the teacher); `provider="modal"` serves the base model
    (bare HF id) or a `modal-lora:adapters/<tag>` adapter on the vLLM app."""
    tag = f"{stage_dir.name}/{env}"
    status = generate_status(stage_dir, env)
    if status == "success" and not force:
        print(f"[{tag}] generate already complete, skipping")
        return
    if status is not None:
        print(f"[{tag}] previous generate ended with status={status}; rerunning")
    cfg = GenerateConfig(
        task=ENV_TASKS[env],
        model=model,
        model_config=ModelConfig(inference_client=InferenceClientConfig(
            provider=provider,
            base_model=BASE_MODEL if provider.startswith("modal") else None,
        )),
        n_samples=N_SAMPLES,
        system_prompts_path=system_prompts_path,
        task_args={**ENV_TASK_ARGS[env], **(extra_task_args or {}), "prompt_ids": prompt_ids},
        max_connections=MAX_CONNECTIONS,
    )
    print(f"[{tag}] generate config:")
    show_config(cfg)
    run_generate(cfg, out=gen_dir(stage_dir, env))


def run_select_stage(
    stage_dir: Path, env: str, method: str, *, filters: list[str] = (), force: bool = False,
) -> None:
    tag = f"{stage_dir.name}/{env}"
    out_path = env_data_path(stage_dir, method, env)
    if out_path.exists() and not force:
        print(f"[{tag}] select already complete ({count_lines(out_path)} rows), skipping")
        return
    if generate_status(stage_dir, env) != "success":
        raise RuntimeError(f"{tag}: generate stage incomplete")
    cfg = SelectConfig(
        input_path=str(gen_dir(stage_dir, env)),
        output_path=str(out_path),
        mode=method,
        seed=SEED,
        filters=list(filters),
        **SELECT_ARGS[method][env],
    )
    print(f"[{tag}] select config:")
    show_config(cfg)
    run_select(cfg, out=stage_dir)


def run_combine_stage(stage_dir: Path, method: str, force: bool = False) -> Path:
    out_path = data_path(stage_dir, method)
    if out_path.exists() and not force:
        print(f"[{stage_dir.name}] combine already complete ({count_lines(out_path)} rows), skipping")
        return out_path
    rows = []
    for env in ENVS:
        text = require(env_data_path(stage_dir, method, env), "select").read_text()
        rows.extend(line for line in text.splitlines() if line.strip())
    if not rows:
        raise RuntimeError(f"{stage_dir.name}: no {method} rows survived selection")
    random.Random(f"{SEED}/combine/{stage_dir.name}").shuffle(rows)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"[{stage_dir.name}] combined {len(rows)} rows -> {out_path}")
    return out_path


def _reattach(stage_dir: Path, info: dict) -> dict:
    """Await a Modal training call spawned by an earlier (interrupted) run of
    this stage rather than resubmitting it."""
    import modal

    print(f"[{stage_dir.name}] re-attaching to Modal call {info['call_id']} ({info['run_tag']})")
    fc = modal.FunctionCall.from_id(info["call_id"])
    result = poll_and_map(
        fc, info["run_tag"], stage_dir, DEFAULT_POLL_HEARTBEAT_S,
        app_name=info["app"], kind=info.get("backend", "modal"), backend=info.get("backend", "modal"),
    )
    return {"model": result.model, "resume_handle": result.resume_handle, "info": result.info}


def run_train_stage(
    stage_dir: Path, method: str, *, prev_adapter: str | None, suffix: str,
    stop_inference: bool = True, force: bool = False,
) -> dict:
    """Submit the stage's Modal LoRA training job and block until it finishes.
    `prev_adapter` (the previous stage's `resume_handle`) continues that
    adapter — LoRA-on-base: every round trains against the same base."""
    result_path = stage_dir / "train_result.json"
    if result_path.exists() and not force:
        result = json.loads(result_path.read_text())
        print(f"[{stage_dir.name}] train already complete, skipping -> {result['model']}")
        return result
    training_file = require(data_path(stage_dir, method), "combine")
    print(f"[{stage_dir.name}] {method} training on {count_lines(training_file)} rows "
          f"(from {prev_adapter or 'base'})")

    info_path = stage_dir / "job_info.json"
    if info_path.exists() and not force:
        result = _reattach(stage_dir, json.loads(info_path.read_text()))
    else:
        if stop_inference:
            # Free this model's inference GPUs for the trainer (the next
            # generation cold-starts the vLLM server again). Kills anything
            # else currently generating against the server, e.g. an eval.
            from rewardhacking_training.modal.modal_utils.inference_utils import (
                stop_server_containers,
            )

            n = stop_server_containers(inference_app_name(BASE_MODEL))
            print(f"[{stage_dir.name}] stopped {n} inference container(s) before training")
        train = train_sft if method == "sft" else train_dpo
        res = train(
            training_file=training_file,
            suffix=suffix,
            prev_adapter_path=prev_adapter,
            base_model_volume_path=f"models/{BASE_MODEL.split('/')[-1]}",
            base_model_repo=BASE_MODEL,
            lora_rank=LORA_RANK,
            micro_batch_size=MICRO_BATCH_SIZE,
            n_gpus=N_GPUS,
            sequence_len=SEQUENCE_LEN,
            wandb_project=WANDB_PROJECT,
            wandb_name=suffix,
            output_dir=stage_dir,
            **TRAIN_HPARAMS[method],
        )
        result = {"model": res.model, "resume_handle": res.resume_handle, "info": res.info}

    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[{stage_dir.name}] trained -> {result['model']}")
    return result


# ---- rounds ---------------------------------------------------------------

def run_sft_round(cond: Condition, *, force: set[str] = frozenset(), stop_inference: bool = True) -> dict:
    """Round 0: best-of-N distillation from the teacher, then the Modal SFT job."""
    d = sft_dir(cond)
    teacher = resolve_model(d, TEACHER_MODEL)
    print(f"\n=== {cond.name} / SFT warmstart: generating from teacher {teacher}")
    for env in ENVS:
        run_generate_stage(
            d, env, teacher, provider="openai",
            system_prompts_path=cond.sft_banks[env],
            extra_task_args=cond.sft_task_args.get(env),
            prompt_ids=round_prompt_ids(env, 0),
            force="generate" in force,
        )
    for env in ENVS:
        run_select_stage(d, env, "sft", filters=cond.sft_filters, force="select" in force)
    run_combine_stage(d, "sft", force="combine" in force)
    return run_train_stage(
        d, "sft", prev_adapter=None, suffix=f"{cond.tag}-sft",
        stop_inference=stop_inference, force="train" in force,
    )


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
    d = iter_dir(cond, i)
    model = resolve_model(d, prev["model"])
    print(f"\n=== {cond.name} / iteration {i}: generating from {model}, "
          f"continuing adapter {prev['resume_handle']}")
    for env in ENVS:
        run_generate_stage(
            d, env, model, provider="modal",
            system_prompts_path=cond.dpo_banks[env],
            prompt_ids=round_prompt_ids(env, i + 1),  # round 0 was the SFT warmstart
            force="generate" in force,
        )
    for env in ENVS:
        run_select_stage(d, env, "dpo", force="select" in force)
    run_combine_stage(d, "dpo", force="combine" in force)
    return run_train_stage(
        d, "dpo", prev_adapter=prev["resume_handle"], suffix=f"{cond.tag}-it{i:02d}",
        stop_inference=stop_inference, force="train" in force,
    )


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
