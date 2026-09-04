"""Shared Modal client helpers for both training engines.

``modal`` (TRL/axolotl) and ``modal_swift`` (ms-swift/Megatron) are two training
engines of the *same* Modal backend — same shared volume, same
``char-vllm-inference-<slug>`` serving app, same ``modal-lora:<path>``
served-name convention, same spawn → heartbeat-poll lifecycle. Everything they
have in common lives here:

* ``resolve_function`` — the per-model app function lookup (with a clear
  deploy-hint error), wrapped by the thin ``_lookup_*`` helpers in ``dpo``/``sft``.
* ``upload_dataset_file`` — push the converted training file to the volume.
* ``poll_and_map`` — the single spawn-result handler: persist ``job_info.json``,
  heartbeat-poll the ``FunctionCall`` to completion, map the payload to a
  ``TrainResult``. (Replaces the three hand-inlined copies of this loop.)
* ``reattach`` — re-join a spawned job from its ``job_info.json`` (resume).
* ``grad_accum_steps`` / ``swift_pipeline_extra_args`` — pure config math.
* ``build_trl_config`` / ``build_swift_config`` — the config-dict bodies shared
  by the DPO and SFT builders (each adds only its objective-specific keys).

All ``modal`` imports are function-scoped so this module imports without the SDK.
"""

from __future__ import annotations

import json
from pathlib import Path

from rewardhacking_training.modal.modal_apps.common import (
    MODAL_LORA_PREFIX,
    VOLUME_MOUNT,
    VOLUME_NAME,
    adapter_dir,
    dataset_path,
    mcore_adapter_dir,
    mcore_model_dir,
)
from rewardhacking_training.train.train_providers.types import TrainResult

DEFAULT_POLL_HEARTBEAT_S = 600
"""`FunctionCall.get` timeout between heartbeat prints while training runs."""


# ---- function lookup + dataset upload -----------------------------------

def resolve_function(app_name: str, fn_name: str, deploy_hint: str, base_model: str):
    """Resolve a function on a per-model app; clear error (with deploy hint) if
    missing. ``base_model`` selects which model's app to look up (its slug must
    match the one the app was deployed with via ``MODAL_TRAIN_BASE_MODEL``)."""
    import modal

    try:
        fn = modal.Function.from_name(app_name, fn_name)
        fn.hydrate()
        return fn
    except Exception as e:  # NotFoundError, auth errors, ...
        raise RuntimeError(
            f"could not resolve Modal function {app_name}/{fn_name} "
            f"({type(e).__name__}: {e}) — {deploy_hint} "
            f"(with MODAL_TRAIN_BASE_MODEL={base_model})"
        ) from e


def upload_dataset_file(local_path: Path, run_tag: str) -> str:
    """Upload a local training JSONL to the shared volume via batch_upload
    (size-robust; no function call). Returns the volume-relative path."""
    import modal

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    remote_rel = dataset_path(run_tag)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), f"/{remote_rel}")
    return remote_rel


# ---- spawn-result handler (the one heartbeat-poll loop) ------------------

def poll_and_map(
    fc, run_tag: str, output_dir: Path | None, poll_heartbeat_s: int,
    *, app_name: str, kind: str, backend: str,
) -> TrainResult:
    """Shared spawn-result handler for every Modal trainer (TRL + swift, DPO +
    SFT): persist ``job_info.json``, heartbeat-poll the Modal ``FunctionCall``
    to completion, and map the payload to a ``TrainResult`` (the ``modal-lora:``
    served-name convention). Raises ``RuntimeError`` on any failure (the
    step-machine halt contract).

    ``kind`` labels the job in log lines; ``backend`` is recorded in
    ``job_info.json``.
    """
    print(f"Modal {kind} job spawned: call_id={fc.object_id} run_tag={run_tag}")
    if output_dir is not None:
        (Path(output_dir) / "job_info.json").write_text(json.dumps({
            "backend": backend,
            "app": app_name,
            "call_id": fc.object_id,
            "run_tag": run_tag,
        }, indent=2))

    import modal.exception as modal_exception

    while True:
        try:
            payload = fc.get(timeout=poll_heartbeat_s)
            break
        except modal_exception.FunctionTimeoutError as e:
            # The REMOTE function exceeded its server-side timeout — fatal.
            raise RuntimeError(
                f"Modal {kind} job {fc.object_id} hit its remote timeout: {e}"
            ) from e
        except (TimeoutError, modal_exception.TimeoutError):
            # Heartbeat: our local fc.get() wait elapsed; job still running.
            # (modal's TimeoutError is NOT a subclass of the builtin.)
            print(f"Modal job {fc.object_id} ({run_tag}): still running")
        except Exception as e:  # remote raise or infra error
            raise RuntimeError(f"Modal {kind} job {fc.object_id} failed: {e}") from e

    if output_dir is not None:
        (Path(output_dir) / "job_final.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )
    adapter_path = payload["adapter_path"]
    return TrainResult(
        model=f"{MODAL_LORA_PREFIX}{adapter_path}",
        resume_handle=adapter_path,
        info=payload,
    )


def reattach(
    info: dict, output_dir: Path, poll_heartbeat_s: int = DEFAULT_POLL_HEARTBEAT_S,
) -> TrainResult:
    """Await the job recorded in ``job_info.json`` (as written by
    ``poll_and_map``) instead of spawning a new one."""
    import modal

    print(f"re-attaching to Modal call {info['call_id']} ({info['run_tag']})")
    backend = info["backend"]
    return poll_and_map(
        modal.FunctionCall.from_id(info["call_id"]), info["run_tag"], output_dir,
        poll_heartbeat_s, app_name=info["app"], kind=backend, backend=backend,
    )


# ---- pure config math ----------------------------------------------------

def grad_accum_steps(batch_size: int, micro_batch_size: int, n_gpus: int) -> int:
    """``batch_size`` is the EFFECTIVE batch; the script's effective batch is
    micro_batch_size x gradient_accumulation_steps x world_size."""
    return max(1, batch_size // max(1, micro_batch_size * n_gpus))


def swift_pipeline_extra_args(
    decoder_first: int | None, decoder_last: int | None,
    extra_args: list[str] | None,
) -> list[str]:
    """Fold the (dash-prefixed) pipeline-balance flags into the megatron
    ``extra_args`` list. Threaded as dedicated ints (not raw extra_args) because
    tyro can't parse ``--``-prefixed values inside a ``list[str]`` CLI arg."""
    out: list[str] = []
    if decoder_first is not None:
        out += ["--decoder_first_pipeline_num_layers", str(decoder_first)]
    if decoder_last is not None:
        out += ["--decoder_last_pipeline_num_layers", str(decoder_last)]
    out += list(extra_args or [])
    return out


# ---- config-dict bodies (each builder adds its objective-specific keys) --

def build_trl_config(
    *,
    run_tag: str,
    base_model_path: str,
    objective: dict,
    lora_rank: int,
    lora_alpha: int | None,
    lora_dropout: float,
    learning_rate: float,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    n_epochs: int,
    lr_scheduler: str,
    warmup_ratio: float,
    weight_decay: float,
    sequence_len: int,
    attn_implementation: str,
    prev_adapter_path: str | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    wandb_project: str | None,
    wandb_name: str | None,
) -> dict:
    """Shared body of the TRL ``build_train_config`` builders. ``objective``
    holds the engine-specific keys (DPO passes ``{"beta": ...}``; SFT passes
    ``{"save_epoch_adapters": ...}``). All volume paths use the in-container
    mount prefix (``/vol``)."""
    cfg: dict = {
        "base_model": f"{VOLUME_MOUNT}/{base_model_path}",
        "dataset_path": f"{VOLUME_MOUNT}/{dataset_path(run_tag)}",
        "output_dir": f"{VOLUME_MOUNT}/{adapter_dir(run_tag)}",
        "run_tag": run_tag,
        **objective,
        # -- LoRA (fp32 adapter on bf16 base; PEFT autocast default) --
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha if lora_alpha is not None else 2 * lora_rank,
        "lora_dropout": lora_dropout,
        # -- optimization --
        "sequence_len": sequence_len,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "attn_implementation": attn_implementation,
        # Quantized base (bitsandbytes) so a large model fits one card under DDP:
        # 4-bit (nf4, QLoRA) or 8-bit (LLM.int8()). Mutually exclusive.
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
    }
    if prev_adapter_path:
        cfg["prev_adapter_path"] = f"{VOLUME_MOUNT}/{prev_adapter_path}"
    if wandb_project:
        cfg["wandb_project"] = wandb_project
        cfg["wandb_name"] = wandb_name or run_tag
    return cfg


def build_swift_config(
    *,
    run_tag: str,
    base_model_repo: str,
    base_model_volume_path: str,
    objective: dict,
    lora_rank: int,
    lora_alpha: int | None,
    learning_rate: float,
    micro_batch_size: int,
    global_batch_size: int,
    n_epochs: int,
    max_length: int,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    expert_model_parallel_size: int,
    expert_tensor_parallel_size: int,
    nproc_per_node: int,
    optimizer_cpu_offload: bool,
    save_steps: int,
    decoder_first_pipeline_num_layers: int | None,
    decoder_last_pipeline_num_layers: int | None,
    prev_mcore_output_dir: str | None,
    extra_args: list[str] | None,
    wandb_project: str | None,
    wandb_name: str | None,
) -> dict:
    """Shared body of the swift ``build_swift_train_config`` builders (`megatron
    sft`/`rlhf` -> `megatron export`). ``objective`` holds the engine-specific
    keys (DPO passes ``{"beta", "rpo_alpha", "loss_type"}``; SFT passes ``{}``).
    All volume paths use the in-container mount prefix (`/vol`)."""
    cfg: dict = {
        "base_model_repo": base_model_repo,
        "mcore_load": f"{VOLUME_MOUNT}/{mcore_model_dir(base_model_volume_path)}",
        "dataset_path": f"{VOLUME_MOUNT}/{dataset_path(run_tag)}",
        "mcore_output_dir": f"{VOLUME_MOUNT}/{mcore_adapter_dir(run_tag)}",
        "hf_adapter_dir": f"{VOLUME_MOUNT}/{adapter_dir(run_tag)}",
        "run_tag": run_tag,
        **objective,
        # LoRA
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha if lora_alpha is not None else 2 * lora_rank,
        "target_modules": "all-linear",
        # parallelism (defaults match the Qwen3-235B-A22B tutorial)
        "tensor_model_parallel_size": tensor_model_parallel_size,
        "pipeline_model_parallel_size": pipeline_model_parallel_size,
        "expert_model_parallel_size": expert_model_parallel_size,
        "expert_tensor_parallel_size": expert_tensor_parallel_size,
        "sequence_parallel": True,
        # optimization
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "num_train_epochs": n_epochs,
        "lr": learning_rate,
        "lr_warmup_fraction": 0.05,
        "min_lr": learning_rate / 10.0,
        "max_length": max_length,
        "optimizer_cpu_offload": optimizer_cpu_offload,
        "save_steps": save_steps,
        "nproc_per_node": nproc_per_node,
        "extra_args": swift_pipeline_extra_args(
            decoder_first_pipeline_num_layers,
            decoder_last_pipeline_num_layers,
            extra_args,
        ),
    }
    if prev_mcore_output_dir:
        cfg["prev_mcore_output_dir"] = f"{VOLUME_MOUNT}/{prev_mcore_output_dir}"
    if wandb_project:
        cfg["wandb_project"] = wandb_project
        cfg["wandb_name"] = wandb_name or run_tag
    return cfg
