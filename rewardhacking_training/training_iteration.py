from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from rewardhacking_training.generate.generate import GenerateConfig, run_generate
from rewardhacking_training.select.select import SelectConfig, run_select
from rewardhacking_training.train.train import TrainConfig, run_train
from rewardhacking_training.train.train_providers.types import write_train_result

Method = Literal["dpo", "sft"]
STAGES = ("generate", "select", "combine", "train")


# ---- layout ---------------------------------------------------------------

def gen_dir(stage_dir: Path, env: str) -> Path:
    return stage_dir / f"generate_{env}"


def env_data_path(stage_dir: Path, method: str, env: str) -> Path:
    return stage_dir / f"{method}_data_{env}.jsonl"


def data_path(stage_dir: Path, method: str) -> Path:
    return stage_dir / f"{method}_data.jsonl"


def generate_status(stage_dir: Path, env: str) -> str | None:
    """Inspect log status of the env's generate stage; None if not run."""
    p = gen_dir(stage_dir, env) / "eval_log.json"
    return json.loads(p.read_text()).get("status") if p.exists() else None


def stage_result(stage_dir: Path) -> dict | None:
    """The iteration's `train_result.json`; None until training has finished."""
    path = stage_dir / "train_result.json"
    return json.loads(path.read_text()) if path.exists() else None


def _require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"{stage} stage incomplete: {path} missing")
    return path


def resolve_model(stage_dir: Path, model: str) -> str:
    """Pinned in `<stage_dir>/model.txt` on first use, so a resumed iteration cannot silently switch checkpoints."""
    pin = stage_dir / "model.txt"
    if pin.exists():
        pinned = pin.read_text().strip()
        if pinned != model:
            raise RuntimeError(
                f"{stage_dir.name} already started from {pinned!r}, not {model!r}"
            )
        return pinned
    stage_dir.mkdir(parents=True, exist_ok=True)
    pin.write_text(model + "\n")
    return model


# ---- stages ---------------------------------------------------------------

def generate_stage(stage_dir: Path, env: str, cfg: GenerateConfig, *, force: bool = False) -> None:
    if generate_status(stage_dir, env) == "success" and not force:
        print(f"[{stage_dir.name}/{env}] generate already complete, skipping")
        return
    run_generate(cfg, out=gen_dir(stage_dir, env))


def select_stage(
    stage_dir: Path, env: str, method: Method, cfg: SelectConfig, *, force: bool = False,
) -> Path:
    """`cfg`'s `mode`, `input_path` and `output_path` are set here from the layout."""
    out_path = env_data_path(stage_dir, method, env)
    if out_path.exists() and not force:
        print(f"[{stage_dir.name}/{env}] select already complete, skipping")
        return out_path
    if generate_status(stage_dir, env) != "success":
        raise RuntimeError(f"{stage_dir.name}/{env}: generate stage incomplete")
    cfg = replace(
        cfg, mode=method, input_path=str(gen_dir(stage_dir, env)), output_path=str(out_path),
    )
    run_select(cfg, out=stage_dir)
    return out_path


def combine_stage(
    stage_dir: Path, method: Method, envs: list[str], *, seed: int, force: bool = False,
) -> Path:
    """Raises if any env's select output is missing or nothing survived."""
    out_path = data_path(stage_dir, method)
    if out_path.exists() and not force:
        print(f"[{stage_dir.name}] combine already complete, skipping")
        return out_path
    rows: list[str] = []
    for env in envs:
        text = _require(env_data_path(stage_dir, method, env), "select").read_text()
        rows.extend(line for line in text.splitlines() if line.strip())
    if not rows:
        raise RuntimeError(f"{stage_dir.name}: no {method} rows survived selection")
    random.Random(f"{seed}/combine/{stage_dir.name}").shuffle(rows)
    out_path.write_text("\n".join(rows) + "\n")
    return out_path


def train_stage(stage_dir: Path, cfg: TrainConfig, *, force: bool = False) -> dict:
    """`training_file` and `output_dir` are set from the layout. An interrupted stage whose job was already
    submitted re-attaches (`run_train` reads `job_info.json`); `force` discards that job and resubmits."""
    result = stage_result(stage_dir)
    if result is not None and not force:
        print(f"[{stage_dir.name}] train already complete, skipping")
        return result
    training_file = _require(data_path(stage_dir, cfg.method), "combine")
    cfg = replace(cfg, training_file=str(training_file), output_dir=str(stage_dir))
    if force:
        (stage_dir / "job_info.json").unlink(missing_ok=True)
    return write_train_result(run_train(cfg), stage_dir)


# ---- one iteration --------------------------------------------------------

@dataclass
class Iteration:
    stage_dir: Path
    method: Method
    model: str
    """The model to generate from (and, on OpenAI, to fine-tune from)."""
    generate: dict[str, GenerateConfig]
    """env -> generate config (with this iteration's `prompt_ids`); `model` is
    set by `run_iteration`."""
    select: dict[str, SelectConfig]
    """env -> select config; `mode` / paths are set by `select_stage`."""
    train: TrainConfig
    """Provider, base model and hyperparameters; `method`, the model /
    checkpoint to continue and the job I/O fields are set by `run_iteration`."""
    suffix: str
    """Job suffix / run tag / W&B run name."""
    resume_handle: str | None = None
    """The previous iteration's checkpoint to continue (None: from the base)."""
    seed: int = 42


def run_iteration(it: Iteration, *, force: set[str] = frozenset()) -> dict:
    """`force` names the stages to redo even if their artifacts exist (`STAGES`)."""
    model = resolve_model(it.stage_dir, it.model)
    envs = list(it.generate)
    for env in envs:
        generate_stage(
            it.stage_dir, env, replace(it.generate[env], model=model), force="generate" in force,
        )
    for env in envs:
        select_stage(it.stage_dir, env, it.method, it.select[env], force="select" in force)
    combine_stage(it.stage_dir, it.method, envs, seed=it.seed, force="combine" in force)
    train_cfg = replace(
        it.train, method=it.method, model=model, resume_handle=it.resume_handle,
        suffix=it.suffix, wandb_name=it.suffix,
    )
    return train_stage(it.stage_dir, train_cfg, force="train" in force)
