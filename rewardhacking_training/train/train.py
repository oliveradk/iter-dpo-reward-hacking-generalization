"""Standalone training driver (DPO or SFT), independent of the iterative loop.

`TrainConfig` is the single, provider-agnostic training config: it promotes the
hyperparameters that are shared across backends (`learning_rate`, `batch_size`,
`n_epochs`, `beta`, `lora_rank`, `lora_alpha`, `wandb_project`) to the top level
and keeps only genuinely provider-specific knobs behind a `<provider>_` prefix.
`run_train(cfg)` submits a single blocking training job and returns a
`TrainResult`; it dispatches on `(cfg.method, cfg.provider)` — DPO supports
openai/tinker/together/modal; SFT supports openai/modal.
When `cfg.output_dir` already holds a `job_info.json` from an interrupted run,
`run_train` re-attaches to that job instead of submitting a new one (openai,
together, modal; tinker trains in-process and always resubmits).

This module knows nothing about the iteration layout; the loop that composes
`run_train` with generate and select lives in
`rewardhacking_training.training_iteration`.

The concrete per-backend `train_dpo` / `train_sft` implementations live in
`rewardhacking_training.train.train_providers`; they are imported lazily inside
each dispatch branch so importing this module never pulls in a backend's heavy
deps (modal, tinker, together, …) unless that backend is actually used.

CLI (one-off training from a merged standardized JSONL):

    python -m rewardhacking_training.train.train <training_file.jsonl> \\
        --provider modal --base-model Qwen/Qwen2.5-32B-Instruct --method dpo \\
        --learning-rate 1e-5 --lora-rank 32 --beta 0.5
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from rewardhacking_training.train.train_providers.types import (
    TrainResult,
    write_train_result,
)

Provider = Literal["openai", "tinker", "together", "modal"]


# ---- config ------------------------------------------------------------

@dataclass
class TrainConfig:
    """Provider-agnostic training config for a single DPO/SFT job.

    Shared hyperparameters (`learning_rate`, `batch_size`, `n_epochs`, `beta`,
    `lora_rank`, `lora_alpha`, `wandb_project`) live at the top level; a knob is
    only kept behind a `<provider>_` prefix when it is genuinely specific to that
    backend. The job-I/O fields (`training_file`, `model`, `resume_handle`,
    `suffix`, `wandb_name`, `output_dir`) describe one concrete run — the
    standalone CLI sets them from flags; the iterative-training wrapper overrides
    them per iteration (the checkpoint/model/output that change each round).
    """

    # ---- identity / method --------------------------------------------
    method: Literal["dpo", "sft"] = "dpo"
    provider: Provider = "openai"
    base_model: str = "gpt-4.1-2025-04-14"

    # ---- job I/O (per-run; set by the CLI or the pipeline wrapper) -----
    training_file: str = ""
    """Standardized DPO/SFT JSONL."""
    model: str | None = None
    """Model to fine-tune from; openai path only (other backends use
    `base_model`/`resume_handle`)."""
    resume_handle: str | None = None
    """Backend checkpoint to continue (tinker state URI / together job id /
    modal adapter path)."""
    suffix: str = ""
    wandb_name: str | None = None
    output_dir: str | None = None

    # ---- shared hyperparameters ---------------------------------------
    learning_rate: float = 1e-5
    """Absolute LR; openai uses `openai_lr_multiplier` instead."""
    batch_size: int | str = 4
    """Effective batch; openai/together accept "auto"/"max", modal/tinker need int."""
    n_epochs: int = 1
    beta: float = 0.5
    """DPO regularization strength (ignored for SFT)."""
    lora_rank: int = 32
    lora_alpha: int | None = None
    """None means 2*lora_rank on modal (server default on together)."""
    wandb_project: str | None = None

    # ---- OpenAI-specific ----------------------------------------------
    openai_lr_multiplier: float | str = 1.0
    openai_poll_interval_s: int | None = None
    """Fixed seconds between job-status polls; None keeps the default backoff."""

    # ---- Tinker-specific ----------------------------------------------
    tinker_model_name: str | None = None
    """Falls back to `base_model` when None."""
    tinker_renderer_name: str | None = None
    tinker_lr_schedule: str = "cosine"
    """`linear`/`cosine`/`constant`; decays after warmup."""
    tinker_warmup_fraction: float = 0.0
    """Fraction of total steps in linear LR warmup (applied by the trainer)."""
    tinker_warmup_steps: int | None = None
    """Fixed number of linear LR warmup steps; overrides `tinker_warmup_fraction`."""
    tinker_save_every: int = 50
    """Checkpoint cadence in steps."""
    tinker_reference_model_name: str | None = None

    # ---- Together-specific --------------------------------------------
    together_lora: bool = True
    together_lora_trainable_modules: str | None = None
    together_rpo_alpha: float | None = None
    """RPO regularization strength (None disables)."""
    together_simpo_gamma: float | None = None
    """SimPO target-margin term (None disables)."""
    together_dpo_normalize_logratios_by_length: bool = False
    together_n_checkpoints: int = 1

    # ---- Modal-specific (LoRA-on-base) ----------------------------------
    modal_micro_batch_size: int = 1
    """Per-device batch; grad-accum is derived from `batch_size`."""
    modal_n_gpus: int = 8
    """World size of the deployed `train` function; drives the grad-accum split."""
    modal_sequence_len: int = 4096
    """Max sequence length."""
    modal_load_in_4bit: bool = False
    """QLoRA (nf4); set `modal_n_gpus=1`."""
    modal_load_in_8bit: bool = False
    modal_base_model_volume_path: str | None = None
    """Volume-relative path; None derives `models/<repo name>`."""
    modal_stop_inference_before_train: bool = False
    """Stop this model's inference containers to free GPUs; costs a vLLM cold
    start every iteration, so leave False unless GPU quota forces it."""


# ---- helpers -----------------------------------------------------------

def _out(cfg: TrainConfig) -> Path | None:
    return Path(cfg.output_dir) if cfg.output_dir else None


def _openai_poll_kwargs(cfg: TrainConfig) -> dict:
    """`poll_schedule_s` override for the openai trainers: a fixed
    `openai_poll_interval_s` becomes a flat single-value schedule (the poll
    loop plateaus at the last value); None keeps the trainer default."""
    if cfg.openai_poll_interval_s is None:
        return {}
    return {"poll_schedule_s": (cfg.openai_poll_interval_s,)}


def _require_int_batch(cfg: TrainConfig, label: str) -> int:
    if not isinstance(cfg.batch_size, int):
        raise RuntimeError(
            f"{label} trainer needs an integer batch_size "
            f"(the effective batch), got {cfg.batch_size!r}"
        )
    return cfg.batch_size


def _modal_pre_train(cfg: TrainConfig) -> str:
    """Shared Modal pre-train setup for both DPO and SFT: optionally free the
    inference server's GPUs, and resolve the base-model volume path. Returns
    the volume-relative base-model path."""
    if cfg.modal_stop_inference_before_train:
        # Serialize GPU use: free THIS model's inference server GPUs (wait
        # until its containers are gone) before the trainer claims them. The
        # next generation cold-starts it again. Scoped to this run's per-model
        # app so a parallel run on a different model is untouched.
        from rewardhacking_training.modal.modal_utils.inference_utils import (
            stop_server_containers,
        )
        from rewardhacking_training.modal.modal_apps.common import (
            inference_app_name,
        )

        n = stop_server_containers(inference_app_name(cfg.base_model))
        print(f"stopped {n} inference container(s) before training")
    return (
        cfg.modal_base_model_volume_path
        or f"models/{cfg.base_model.split('/')[-1]}"
    )


# ---- SFT dispatch ------------------------------------------------------

def _dispatch_train_sft(cfg: TrainConfig) -> TrainResult:
    """SFT training dispatch on `cfg.provider` (openai, modal).
    LoRA-on-base resume mirrors the DPO path: every Modal iteration trains
    against `base_model`, continuing the prior adapter via `resume_handle`."""
    if cfg.provider == "openai":
        from rewardhacking_training.train.train_providers.openai.sft import (
            train_sft as openai_train_sft,
        )
        return openai_train_sft(
            training_file=Path(cfg.training_file),
            model=cfg.model,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            learning_rate_multiplier=cfg.openai_lr_multiplier,
            suffix=cfg.suffix,
            output_dir=_out(cfg),
            **_openai_poll_kwargs(cfg),
        )
    if cfg.provider == "modal":
        from rewardhacking_training.train.train_providers.modal.sft import (
            train_sft as modal_train_sft,
        )
        batch_size = _require_int_batch(cfg, "modal")
        return modal_train_sft(
            training_file=Path(cfg.training_file),
            suffix=cfg.suffix,
            prev_adapter_path=cfg.resume_handle,
            base_model_volume_path=_modal_pre_train(cfg),
            base_model_repo=cfg.base_model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            learning_rate=cfg.learning_rate,
            batch_size=batch_size,
            micro_batch_size=cfg.modal_micro_batch_size,
            n_gpus=cfg.modal_n_gpus,
            n_epochs=cfg.n_epochs,
            sequence_len=cfg.modal_sequence_len,
            load_in_4bit=cfg.modal_load_in_4bit,
            load_in_8bit=cfg.modal_load_in_8bit,
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            output_dir=_out(cfg),
        )
    raise RuntimeError(
        f"method='sft' is only supported for provider in (openai, modal); "
        f"got provider={cfg.provider!r}"
    )


# ---- DPO dispatch ------------------------------------------------------

def _dispatch_train_dpo(cfg: TrainConfig) -> TrainResult:
    """DPO training dispatch on `cfg.provider` (openai, tinker, together,
    modal)."""
    if cfg.provider == "openai":
        from rewardhacking_training.train.train_providers.openai.dpo import (
            train_dpo as openai_train_dpo,
        )
        return openai_train_dpo(
            training_file=Path(cfg.training_file),
            model=cfg.model,
            beta=cfg.beta,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            learning_rate_multiplier=cfg.openai_lr_multiplier,
            suffix=cfg.suffix,
            output_dir=_out(cfg),
            **_openai_poll_kwargs(cfg),
        )
    if cfg.provider == "tinker":
        from rewardhacking_training.train.train_providers.tinker.dpo import (
            train_dpo as tinker_train_dpo,
        )
        log_dir = _out(cfg) or Path(".")
        return tinker_train_dpo(
            training_file=Path(cfg.training_file),
            model_name=cfg.tinker_model_name or cfg.base_model,
            log_path=log_dir / "tinker_log",
            beta=cfg.beta,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size if isinstance(cfg.batch_size, int) else None,
            learning_rate=cfg.learning_rate,
            lr_schedule=cfg.tinker_lr_schedule,
            warmup_fraction=cfg.tinker_warmup_fraction,
            warmup_steps=cfg.tinker_warmup_steps,
            lora_rank=cfg.lora_rank,
            save_every=cfg.tinker_save_every,
            load_checkpoint_path=cfg.resume_handle,
            reference_model_name=cfg.tinker_reference_model_name,
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            renderer_name=cfg.tinker_renderer_name,
        )
    if cfg.provider == "together":
        from rewardhacking_training.train.train_providers.together.dpo import (
            train_dpo as together_train_dpo,
        )
        # Checkpoint-resumable: with no `resume_handle` we fine-tune the
        # trainable base model; otherwise continue from the prior job via
        # `from_checkpoint` (Together's `create` takes a base in `model`, never
        # a served fine-tune name).
        resume = cfg.resume_handle
        return together_train_dpo(
            training_file=Path(cfg.training_file),
            model=None if resume else cfg.base_model,
            from_checkpoint=resume,
            beta=cfg.beta,
            n_epochs=cfg.n_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            lora=cfg.together_lora,
            lora_r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_trainable_modules=cfg.together_lora_trainable_modules,
            rpo_alpha=cfg.together_rpo_alpha,
            simpo_gamma=cfg.together_simpo_gamma,
            dpo_normalize_logratios_by_length=(
                cfg.together_dpo_normalize_logratios_by_length
            ),
            n_checkpoints=cfg.together_n_checkpoints,
            suffix=cfg.suffix,
            wandb_project_name=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            output_dir=_out(cfg),
        )
    if cfg.provider == "modal":
        from rewardhacking_training.train.train_providers.modal.dpo import (
            train_dpo as modal_train_dpo,
        )
        # LoRA-on-base: train against `base_model` on the shared volume;
        # continue the previous adapter (`resume_handle` = its volume-relative
        # path) when set.
        batch_size = _require_int_batch(cfg, "modal")
        return modal_train_dpo(
            training_file=Path(cfg.training_file),
            suffix=cfg.suffix,
            prev_adapter_path=cfg.resume_handle,
            base_model_volume_path=_modal_pre_train(cfg),
            base_model_repo=cfg.base_model,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            learning_rate=cfg.learning_rate,
            beta=cfg.beta,
            batch_size=batch_size,
            micro_batch_size=cfg.modal_micro_batch_size,
            n_gpus=cfg.modal_n_gpus,
            n_epochs=cfg.n_epochs,
            sequence_len=cfg.modal_sequence_len,
            load_in_4bit=cfg.modal_load_in_4bit,
            load_in_8bit=cfg.modal_load_in_8bit,
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            output_dir=_out(cfg),
        )
    raise RuntimeError(f"unsupported DPO provider: {cfg.provider!r}")


# ---- entrypoint --------------------------------------------------------

def _reattach(cfg: TrainConfig) -> TrainResult | None:
    """Await the job recorded in `<output_dir>/job_info.json` by an earlier
    (interrupted) `run_train` of this config, rather than resubmitting.
    Returns None when there is no such job or the provider cannot re-attach
    (tinker), in which case the caller submits afresh."""
    out = _out(cfg)
    info_path = out / "job_info.json" if out is not None else None
    if info_path is None or not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    if cfg.provider == "openai" and "id" in info:
        from rewardhacking_training.train.train_providers.openai.utils import reattach

        return reattach(info, out, **_openai_poll_kwargs(cfg))
    if cfg.provider == "together" and "id" in info:
        from rewardhacking_training.train.train_providers.together.utils import reattach

        return reattach(info, out)
    if cfg.provider == "modal" and "call_id" in info:
        from rewardhacking_training.train.train_providers.modal.utils import reattach

        return reattach(info, out)
    return None


def run_train(cfg: TrainConfig) -> TrainResult:
    """Run a single training job (DPO or SFT) and block until it finishes.

    Dispatches on `(cfg.method, cfg.provider)`: DPO supports
    openai/tinker/together/modal; SFT supports openai/modal. If
    `cfg.output_dir` holds a `job_info.json` from an
    interrupted run, the job it names is awaited instead of submitting a new
    one (delete the file to force a resubmit). Raises `RuntimeError` on
    submission/validation failure or an unsupported `(method, provider)`
    combination; callers that want to keep running should catch it.
    """
    result = _reattach(cfg)
    if result is not None:
        return result
    if cfg.method == "sft":
        return _dispatch_train_sft(cfg)
    return _dispatch_train_dpo(cfg)


# ---- CLI ---------------------------------------------------------------

def main():
    load_dotenv()
    import tyro

    argv = sys.argv[1:]
    training_file: str | None = None
    if argv and not argv[0].startswith("-"):
        training_file, *argv = argv

    cfg = tyro.cli(TrainConfig, args=argv)
    if training_file:
        cfg = replace(cfg, training_file=training_file)
    if not cfg.training_file:
        raise SystemExit(
            "training_file is required (positional path or --training-file)"
        )
    if not cfg.output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = replace(cfg, output_dir=f"output/train/{cfg.provider}_{ts}")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    result = run_train(cfg)

    write_train_result(result, out)
    print(f"model={result.model}")
    print(f"Run dir: {out}")


if __name__ == "__main__":
    main()
