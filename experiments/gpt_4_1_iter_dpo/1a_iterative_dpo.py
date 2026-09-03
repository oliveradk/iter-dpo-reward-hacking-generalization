from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

from common import BASE_MODEL, REPO_ROOT, RUN_DIR, RUN_NAME

assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

from rewardhacking_training.generate.generate import (
    GenerateConfig,
    ModelConfig,
    _resolve_task,
    run_generate,
)
from rewardhacking_training.generate.inference_client import (
    InferenceClientConfig,
)
from rewardhacking_training.select.select import SelectConfig, run_select
from rewardhacking_training.train.train_providers.openai.dpo import (
    convert_standardized_file_to_openai,
    submit_dpo_job,
)
from rewardhacking_training.train.train_providers.openai.utils import (
    get_client,
    save_job_info,
    upload_training_file,
)

# ---- hyperparameters (paper gpt-4.1 recipe) -------------------------------
# BASE_MODEL / RUN_NAME / RUN_DIR come from `common` so the eval scripts can
# find this run's checkpoints.

SEED = 42

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


def gen_dir(i: int, env: str) -> Path:
    return iter_dir(i) / f"generate_{env}"


def env_dpo_path(i: int, env: str) -> Path:
    return iter_dir(i) / f"dpo_data_{env}.jsonl"


def infer_iteration() -> int:
    """First iteration whose job has not been submitted yet."""
    i = 0
    while (iter_dir(i) / "job_info.json").exists():
        i += 1
    return i


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open())


# ---- manual epoch indexing ------------------------------------------------

def iteration_prompt_ids(env: str, i: int) -> list[str]:
    """Iteration `i`'s chunk of env's prompt stream (independently shuffled
    epochs, concatenated), `EPOCH_FRACTION[env]` of an epoch per iteration."""
    task = _resolve_task(ENV_TASKS[env])(**ENV_TASK_ARGS[env])
    epoch_ids = [str(s.id) for s in task.dataset]
    size = len(epoch_ids)
    count = max(1, round(EPOCH_FRACTION[env] * size))
    start, stop = i * count, (i + 1) * count
    ids: list[str] = []
    for epoch in range(start // size, math.ceil(stop / size)):
        order = list(epoch_ids)
        random.Random(f"{SEED}/{env}/epoch/{epoch}").shuffle(order)
        ids.extend(order[max(start - epoch * size, 0):stop - epoch * size])
    print(f"[iter {i}/{env}] epoch size {size}; {len(ids)} prompts from stream offset {start}")
    return ids


# ---- resumable stages -----------------------------------------------------
# Each stage skips itself when its artifact exists (unless forced), and
# raises when the previous stage's artifact is missing.

def show_config(cfg) -> None:
    for k, v in asdict(cfg).items():
        if k == "task_args" and "prompt_ids" in v:
            v = {**v, "prompt_ids": f"[{len(v['prompt_ids'])} ids]"}
        print(f"  {k} = {v}")


def require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{stage} stage incomplete: {path} missing")
    return path


def generate_status(i: int, env: str) -> str | None:
    p = gen_dir(i, env) / "eval_log.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def run_generate_stage(i: int, env: str, model: str, force: bool = False) -> None:
    status = generate_status(i, env)
    if status == "success" and not force:
        print(f"[iter {i}/{env}] generate already complete, skipping")
        return
    if status is not None:
        print(f"[iter {i}/{env}] previous generate ended with status={status}; rerunning")
    cfg = GenerateConfig(
        task=ENV_TASKS[env],
        model=model,  # bare / ft: OpenAI id; the inference client prefixes openai/
        model_config=ModelConfig(inference_client=InferenceClientConfig(provider="openai")),
        n_samples=N_SAMPLES[env],
        system_prompts_path=SYSTEM_PROMPTS_PATH,
        task_args={**ENV_TASK_ARGS[env], "prompt_ids": iteration_prompt_ids(env, i)},
        max_connections=MAX_CONNECTIONS,
    )
    print(f"[iter {i}/{env}] generate config:")
    show_config(cfg)
    run_generate(cfg, out=gen_dir(i, env))


def run_select_stage(i: int, env: str, force: bool = False) -> None:
    out_path = env_dpo_path(i, env)
    if out_path.exists() and not force:
        print(f"[iter {i}/{env}] select already complete ({count_lines(out_path)} rows), skipping")
        return
    if generate_status(i, env) != "success":
        raise RuntimeError(f"iter {i}/{env}: generate stage incomplete")
    cfg = SelectConfig(
        input_path=str(gen_dir(i, env)),
        output_path=str(out_path),
        mode="dpo",
        seed=SEED,
        **ENV_SELECT_ARGS[env],
    )
    print(f"[iter {i}/{env}] select config:")
    show_config(cfg)
    run_select(cfg, out=iter_dir(i))


def run_combine_stage(i: int, force: bool = False) -> None:
    out_path = iter_dir(i) / "dpo_data.jsonl"
    if out_path.exists() and not force:
        print(f"[iter {i}] combine already complete ({count_lines(out_path)} rows), skipping")
        return
    rows = []
    for env in ENVS:
        text = require(env_dpo_path(i, env), "select").read_text()
        rows.extend(line for line in text.splitlines() if line.strip())
    random.Random(f"{SEED}/combine/{i}").shuffle(rows)
    out_path.write_text("\n".join(rows) + "\n")
    print(f"[iter {i}] combined {len(rows)} rows -> {out_path}")


def run_train_stage(i: int, model: str, force: bool = False) -> dict:
    """Upload the combined file and submit the OpenAI DPO job; does NOT poll."""
    info_path = iter_dir(i) / "job_info.json"
    if info_path.exists() and not force:
        info = json.loads(info_path.read_text())
        print(f"[iter {i}] job already submitted, skipping -> {info['id']}")
        return info
    dpo_path = require(iter_dir(i) / "dpo_data.jsonl", "combine")
    openai_path = convert_standardized_file_to_openai(dpo_path, dpo_path.with_suffix(".openai.jsonl"))
    client = get_client()
    file_id = upload_training_file(client, openai_path)
    info = submit_dpo_job(
        client, file_id, model,
        beta=DPO_BETA, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
        learning_rate_multiplier=LR_MULTIPLIER,
        suffix=f"{RUN_NAME}-it{i:02d}".replace("_", "-")[:40],
    )
    save_job_info(info, iter_dir(i))
    print(f"[iter {i}] OpenAI DPO job submitted: {info['id']} (from {model})")
    return info


# ---- main ----------------------------------------------------------------

STAGES = ["generate", "select", "combine", "train"]


def resolve_model(i: int, requested: str | None) -> str:
    """The model iteration `i` generates from and fine-tunes. Pinned in
    `iter_NN/model.txt` on first use so a resumed iteration can't switch."""
    pin = iter_dir(i) / "model.txt"
    pinned = pin.read_text().strip() if pin.exists() else None
    model = requested or pinned or (BASE_MODEL if i == 0 else None)
    if model is None:
        sys.exit(f"iteration {i} needs --model <ft: checkpoint from iteration {i - 1}>")
    if pinned and pinned != model:
        sys.exit(f"iteration {i} already started from {pinned!r}, not {model!r}")
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(model + "\n")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="checkpoint to generate from and fine-tune "
                    "(previous iteration's ft: id); defaults to the base model at iteration 0")
    ap.add_argument("--iteration", type=int, help="default: first iteration without a submitted job")
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES,
                    help="redo these stages even if their artifacts exist")
    args = ap.parse_args()

    i = infer_iteration() if args.iteration is None else args.iteration
    model = resolve_model(i, args.model)
    force = set(args.force)
    print(f"run dir: {RUN_DIR}\niteration {i}, model {model}")

    for env in ENVS:
        run_generate_stage(i, env, model, force="generate" in force)
    for env in ENVS:
        run_select_stage(i, env, force="select" in force)
    run_combine_stage(i, force="combine" in force)
    info = run_train_stage(i, model, force="train" in force)

    print(
        f"\nJob {info['id']} submitted; this script does not wait for it. Check with:\n"
        f"  python -c \"import openai; print(openai.OpenAI().fine_tuning.jobs.retrieve('{info['id']}').fine_tuned_model)\"\n"
        f"then run iteration {i + 1} with:\n"
        f"  python {Path(__file__).relative_to(REPO_ROOT)} --model <fine_tuned_model>"
    )


if __name__ == "__main__":
    main()
