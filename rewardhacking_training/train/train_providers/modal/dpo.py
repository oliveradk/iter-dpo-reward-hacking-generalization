"""Modal DPO trainer (TRL + ms-swift/Megatron).

A remote job-based backend like the OpenAI/Together trainers, but the "job API"
is our own Modal app: convert the standardized DPO JSONL to the trainer's native
format, upload it to the shared volume, spawn the remote ``train`` function with
a client-built config, and block on the call with a heartbeat poll (all via
``modal.utils``). Two engines live here:

* TRL (``train_dpo``, app ``char-dpo-train-<slug>``) — the default. Converts to
  the input/chosen/rejected ("icr") format; the remote side runs
  ``trl_dpo_script.py`` (not ``axolotl train``, whose RL path can't continue a
  prior LoRA adapter).
* ms-swift/Megatron (``train_dpo_swift``, app ``char-swift-dpo-train-<slug>``) —
  for very large MoE bases (e.g. Qwen3-235B-A22B). Converts to the ms-swift DPO
  ``messages`` format; drives ``megatron rlhf --rlhf_type dpo``.

Both use **LoRA-on-base** iteration (every iteration trains against the same
base; iter >= 1 continues the prior adapter) and return a ``TrainResult`` whose
``model`` is ``modal-lora:<adapter path>`` and ``resume_handle`` the bare
adapter path.

Deployment is manual (``modal deploy …``); lookup failures raise a RuntimeError
saying exactly which app to deploy. All ``modal`` imports are function-scoped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rewardhacking_training.modal.modal_apps.common import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_BASE_MODEL_REPO,
    mcore_adapter_dir,
    sanitize_run_tag,
    swift_dpo_train_app_name,
    train_app_name,
)
from rewardhacking_training.train.train_providers.modal.utils import (
    DEFAULT_POLL_HEARTBEAT_S,
    build_swift_config,
    build_trl_config,
    grad_accum_steps,
    poll_and_map,
    resolve_function,
    upload_dataset_file,
)
from rewardhacking_training.train.train_providers.types import (
    TrainResult,
    assistant_content,
    convert_file,
    split_system_user,
    write_train_result,
)

__all__ = [
    "standardized_pair_to_icr", "convert_standardized_file_to_icr",
    "grad_accum_steps", "build_train_config", "train_dpo", "merge_adapter",
    "standardized_pair_to_messages", "convert_standardized_file_to_messages",
    "build_swift_train_config", "train_dpo_swift",
]

_DEPLOY_HINT = (
    "run: modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_dpo_app.py"
)
_SWIFT_DEPLOY_HINT = (
    "run: modal deploy rewardhacking_training/modal/modal_apps/modal_swift_train/modal_dpo_swift_app.py"
)


def _lookup_function(name: str, base_model: str = DEFAULT_BASE_MODEL_REPO):
    """Resolve a function on the per-model ``char-dpo-train-<slug>`` app."""
    return resolve_function(train_app_name(base_model), name, _DEPLOY_HINT, base_model)


def _lookup_swift_function(name: str, base_model: str):
    """Resolve a function on the per-model ``char-swift-dpo-train-<slug>`` app."""
    return resolve_function(
        swift_dpo_train_app_name(base_model), name, _SWIFT_DEPLOY_HINT, base_model
    )


# ---- standardized-format -> icr conversion (TRL) ------------------------

def standardized_pair_to_icr(row: dict, *, use_full_response: bool = True) -> dict:
    """Convert a standardized DPO row (see ``types``) to the input/chosen/
    rejected JSONL row the TRL script consumes:

        {"system": ..., "input": ..., "chosen": ..., "rejected": ...}

    The prompt field is ``input``; ``system`` is included only when the source
    row has a system message. Values are BARE strings — the script renders the
    chat template and appends EOS server-side. Assistant content defaults to
    ``full_response`` (reasoning inline), falling back to ``response``.
    """
    system, user = split_system_user(row["input"]["messages"])
    if user is None:
        raise ValueError(f"standardized DPO row has no user message: {row['input']!r}")
    out: dict = {
        "input": user,
        "chosen": assistant_content(row["preferred_output"], use_full_response),
        "rejected": assistant_content(row["non_preferred_output"], use_full_response),
    }
    if system:
        out["system"] = system
    return out


def convert_standardized_file_to_icr(
    in_path: Path, out_path: Path, *, use_full_response: bool = True
) -> Path:
    """Read a standardized DPO JSONL and write the icr equivalent."""
    return convert_file(
        in_path, out_path,
        lambda r: standardized_pair_to_icr(r, use_full_response=use_full_response),
    )


# ---- training-config builder (TRL; consumed by trl_dpo_script.py) -------

def build_train_config(
    *,
    run_tag: str,
    base_model_path: str = DEFAULT_BASE_MODEL_PATH,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    learning_rate: float = 1e-5,
    beta: float = 0.1,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    n_epochs: int = 1,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    sequence_len: int = 4096,
    attn_implementation: str = "flash_attention_2",
    prev_adapter_path: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> dict:
    """Build the config dict the remote ``train`` function hands to
    ``trl_dpo_script.py``. Adds the DPO ``beta`` to the shared TRL config body
    (see ``modal.utils.build_trl_config``).

    ``prev_adapter_path`` (volume-relative) makes iter >= 1 load the previous
    adapter with ``is_trainable=True`` (LoRA-on-base continuation).
    """
    return build_trl_config(
        run_tag=run_tag,
        base_model_path=base_model_path,
        objective={"beta": beta},
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        n_epochs=n_epochs,
        lr_scheduler=lr_scheduler,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        sequence_len=sequence_len,
        attn_implementation=attn_implementation,
        prev_adapter_path=prev_adapter_path,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


# ---- blocking entrypoint (TRL DPO) --------------------------------------

def train_dpo(
    *,
    training_file: Path,
    suffix: str,
    prev_adapter_path: str | None = None,
    base_model_volume_path: str = DEFAULT_BASE_MODEL_PATH,
    base_model_repo: str = DEFAULT_BASE_MODEL_REPO,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    learning_rate: float = 1e-5,
    beta: float = 0.1,
    batch_size: int = 4,
    micro_batch_size: int = 1,
    n_gpus: int = 8,
    n_epochs: int = 1,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    sequence_len: int = 4096,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    output_dir: Path | None = None,
    use_full_response: bool = True,
    poll_heartbeat_s: int = DEFAULT_POLL_HEARTBEAT_S,
) -> TrainResult:
    """Submit + block on a Modal TRL DPO job.

    ``training_file`` is a standardized DPO JSONL (see ``types``); it is
    converted to the icr format before upload. ``suffix`` becomes the run tag
    (sanitized) and names every volume artifact. ``batch_size`` is the EFFECTIVE
    batch — split into ``micro_batch_size`` x grad-accum x ``n_gpus``. Set
    ``prev_adapter_path`` (from the prior ``TrainResult.resume_handle``) to
    continue training the previous iteration's adapter (LoRA-on-base resume).

    Persists ``job_info.json`` on spawn (incl. the Modal call id, so a crashed
    client can re-attach) and ``job_final.json`` on completion when
    ``output_dir`` is set. Raises ``RuntimeError`` on any failure (halt
    contract).
    """
    run_tag = sanitize_run_tag(suffix)
    if not run_tag:
        raise RuntimeError(f"suffix {suffix!r} sanitized to an empty run tag")
    training_file = Path(training_file)
    icr_file = training_file.with_suffix(".icr.jsonl")

    # Convert + upload + spawn are the deterministic-failure surface; wrap
    # everything into RuntimeError per the step-machine halt contract.
    try:
        convert_standardized_file_to_icr(
            training_file, icr_file, use_full_response=use_full_response
        )
        upload_dataset_file(icr_file, run_tag)

        # Idempotent fast no-op once the snapshot exists on the volume.
        download_fn = _lookup_function("download_base_model", base_model_repo)
        download_fn.remote(repo_id=base_model_repo, dest=base_model_volume_path)

        cfg = build_train_config(
            run_tag=run_tag,
            base_model_path=base_model_volume_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            beta=beta,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=grad_accum_steps(
                batch_size, micro_batch_size, n_gpus
            ),
            n_epochs=n_epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            sequence_len=sequence_len,
            prev_adapter_path=prev_adapter_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
        )
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "train_config.json").write_text(
                json.dumps(cfg, indent=2)
            )

        train_fn = _lookup_function("train", base_model_repo)
        fc = train_fn.spawn(cfg, run_tag)
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize to the step-machine contract
        raise RuntimeError(f"Modal DPO submission failed: {e}") from e

    return poll_and_map(
        fc, run_tag, output_dir, poll_heartbeat_s,
        app_name=train_app_name(base_model_repo), kind="DPO", backend="modal",
    )


def merge_adapter(
    adapter_path: str, run_tag: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL_REPO,
) -> dict:
    """Standalone merge: fold a volume-resident LoRA adapter into its base model
    and save under ``merged/<run_tag>``. Returns the remote payload
    (``{"merged_path": ...}``). ``base_model`` selects the per-model app that
    runs the merge (the adapter's own base is read remotely from its config)."""
    merge_fn = _lookup_function("merge", base_model)
    return merge_fn.remote(adapter_path=adapter_path, run_tag=run_tag)


# ---- ms-swift / Megatron DPO (provider="modal_swift") --------------------

def standardized_pair_to_messages(row: dict, *, use_full_response: bool = True) -> dict:
    """Convert a standardized DPO row to the ms-swift DPO JSONL format:

        {"messages": [{system?}, {user}, {assistant = <chosen>}],
         "rejected_response": <rejected>}

    The prompt messages are preserved; the PREFERRED completion is appended as
    the final assistant turn and the NON-PREFERRED completion goes in
    ``rejected_response``. Completion content defaults to ``full_response``
    (reasoning inline), falling back to ``response``."""
    messages = [dict(m) for m in row["input"]["messages"]]
    if not any(m["role"] == "user" for m in messages):
        raise ValueError(f"standardized DPO row has no user message: {messages!r}")
    messages.append({
        "role": "assistant",
        "content": assistant_content(row["preferred_output"], use_full_response),
    })
    return {
        "messages": messages,
        "rejected_response": assistant_content(
            row["non_preferred_output"], use_full_response
        ),
    }


def convert_standardized_file_to_messages(
    in_path: Path, out_path: Path, *, use_full_response: bool = True
) -> Path:
    """Read a standardized DPO JSONL and write the ms-swift DPO `messages`
    JSONL."""
    return convert_file(
        in_path, out_path,
        lambda r: standardized_pair_to_messages(r, use_full_response=use_full_response),
    )


def build_swift_train_config(
    *,
    run_tag: str,
    base_model_repo: str,
    base_model_volume_path: str,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    learning_rate: float = 1e-5,
    beta: float = 0.1,
    rpo_alpha: float | None = None,
    micro_batch_size: int = 1,
    global_batch_size: int = 8,
    n_epochs: int = 1,
    max_length: int = 2048,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 4,
    expert_model_parallel_size: int = 2,
    expert_tensor_parallel_size: int = 1,
    nproc_per_node: int = 8,
    optimizer_cpu_offload: bool = True,
    save_steps: int = 50,
    decoder_first_pipeline_num_layers: int | None = None,
    decoder_last_pipeline_num_layers: int | None = None,
    prev_mcore_output_dir: str | None = None,
    extra_args: list[str] | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> dict:
    """Build the config dict the swift DPO app's ``train`` hands to
    ``swift_dpo_script.py`` (`megatron rlhf --rlhf_type dpo` -> `megatron
    export`). Adds the DPO ``beta`` / ``rpo_alpha`` / ``loss_type`` to the shared
    swift config body (see ``modal.utils.build_swift_config``). Swift DPO
    defaults to **pure DPO** (``rpo_alpha=None``)."""
    return build_swift_config(
        run_tag=run_tag,
        base_model_repo=base_model_repo,
        base_model_volume_path=base_model_volume_path,
        objective={"beta": beta, "rpo_alpha": rpo_alpha, "loss_type": "sigmoid"},
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        n_epochs=n_epochs,
        max_length=max_length,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        nproc_per_node=nproc_per_node,
        optimizer_cpu_offload=optimizer_cpu_offload,
        save_steps=save_steps,
        decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
        decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
        prev_mcore_output_dir=prev_mcore_output_dir,
        extra_args=extra_args,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


def train_dpo_swift(
    *,
    training_file: Path,
    suffix: str,
    prev_adapter_path: str | None = None,
    base_model_volume_path: str = DEFAULT_BASE_MODEL_PATH,
    base_model_repo: str = DEFAULT_BASE_MODEL_REPO,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    learning_rate: float = 1e-5,
    beta: float = 0.1,
    rpo_alpha: float | None = None,
    global_batch_size: int = 8,
    micro_batch_size: int = 1,
    n_epochs: int = 1,
    max_length: int = 2048,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 4,
    expert_model_parallel_size: int = 2,
    decoder_first_pipeline_num_layers: int | None = None,
    decoder_last_pipeline_num_layers: int | None = None,
    nproc_per_node: int = 8,
    optimizer_cpu_offload: bool = True,
    save_steps: int = 50,
    extra_megatron_args: list[str] | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    output_dir: Path | None = None,
    use_full_response: bool = True,
    poll_heartbeat_s: int = DEFAULT_POLL_HEARTBEAT_S,
) -> TrainResult:
    """Submit + block on a Modal ms-swift/Megatron LoRA DPO job for a large MoE
    model. Converts the standardized DPO JSONL to the ms-swift DPO `messages`
    format, uploads it, ensures the HF base + its mcore conversion are on the
    volume, spawns ``train``, and heartbeat-polls to completion.

    ``prev_adapter_path`` (the prior iteration's HF adapter path) continues that
    iteration's mcore LoRA adapter (LoRA-on-base). Returns a ``TrainResult``
    whose ``model`` is ``modal-lora:<hf adapter path>`` and ``resume_handle`` the
    bare HF adapter path (identical convention to the TRL modal path)."""
    run_tag = sanitize_run_tag(suffix)
    if not run_tag:
        raise RuntimeError(f"suffix {suffix!r} sanitized to an empty run tag")
    training_file = Path(training_file)
    messages_file = training_file.with_suffix(".messages.jsonl")

    try:
        convert_standardized_file_to_messages(
            training_file, messages_file, use_full_response=use_full_response
        )
        upload_dataset_file(messages_file, run_tag)

        download_fn = _lookup_swift_function("download_base_model", base_model_repo)
        download_fn.remote(repo_id=base_model_repo, dest=base_model_volume_path)
        convert_fn = _lookup_swift_function("convert_to_mcore", base_model_repo)
        convert_fn.remote(base_model_volume_path=base_model_volume_path)

        prev_mcore_output_dir = (
            mcore_adapter_dir(sanitize_run_tag(Path(prev_adapter_path).name))
            if prev_adapter_path else None
        )
        cfg = build_swift_train_config(
            run_tag=run_tag,
            base_model_repo=base_model_repo,
            base_model_volume_path=base_model_volume_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
            beta=beta,
            rpo_alpha=rpo_alpha,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            n_epochs=n_epochs,
            max_length=max_length,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            decoder_first_pipeline_num_layers=decoder_first_pipeline_num_layers,
            decoder_last_pipeline_num_layers=decoder_last_pipeline_num_layers,
            nproc_per_node=nproc_per_node,
            optimizer_cpu_offload=optimizer_cpu_offload,
            save_steps=save_steps,
            prev_mcore_output_dir=prev_mcore_output_dir,
            extra_args=extra_megatron_args,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
        )
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "swift_train_config.json").write_text(
                json.dumps(cfg, indent=2)
            )

        train_fn = _lookup_swift_function("train", base_model_repo)
        fc = train_fn.spawn(cfg, run_tag)
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize to the step-machine contract
        raise RuntimeError(f"Modal swift DPO submission failed: {e}") from e

    return poll_and_map(
        fc, run_tag, output_dir, poll_heartbeat_s,
        app_name=swift_dpo_train_app_name(base_model_repo),
        kind="swift-dpo", backend="modal-swift-dpo",
    )


# ---- CLI ---------------------------------------------------------------

@dataclass
class ModalDPOJobConfig:
    training_file: str | None = None
    suffix: str = "modal-dpo"
    prev_adapter_path: str | None = None
    base_model_volume_path: str = DEFAULT_BASE_MODEL_PATH
    base_model_repo: str = DEFAULT_BASE_MODEL_REPO
    lora_rank: int = 32
    lora_alpha: int | None = None
    learning_rate: float = 1e-5
    beta: float = 0.1
    batch_size: int = 4
    micro_batch_size: int = 1
    n_gpus: int = 8
    """World size of the deployed train app (grad-accum math only — the actual
    GPU count is fixed at deploy time via MODAL_TRAIN_GPU)."""
    n_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    sequence_len: int = 4096
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    wandb_project: str | None = None
    wandb_name: str | None = None
    output_dir: str | None = None
    use_full_response: bool = True
    merge_adapter_path: str | None = None
    """Set to run a standalone merge of an existing adapter instead of training."""


def main():
    import tyro
    from dotenv import load_dotenv

    load_dotenv()
    cfg = tyro.cli(ModalDPOJobConfig)
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(cfg), indent=2, default=str)
        )

    if cfg.merge_adapter_path:
        payload = merge_adapter(
            cfg.merge_adapter_path, run_tag=cfg.suffix,
            base_model=cfg.base_model_repo,
        )
        print(f"Merged: {payload}")
        if output_dir:
            (output_dir / "merge_result.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )
        return payload

    if not cfg.training_file:
        raise SystemExit("--training-file is required (unless --merge-adapter-path)")
    result = train_dpo(
        training_file=Path(cfg.training_file),
        suffix=cfg.suffix,
        prev_adapter_path=cfg.prev_adapter_path,
        base_model_volume_path=cfg.base_model_volume_path,
        base_model_repo=cfg.base_model_repo,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        batch_size=cfg.batch_size,
        micro_batch_size=cfg.micro_batch_size,
        n_gpus=cfg.n_gpus,
        n_epochs=cfg.n_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        sequence_len=cfg.sequence_len,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        output_dir=output_dir,
        use_full_response=cfg.use_full_response,
    )
    print(f"Modal DPO done: model={result.model} resume_handle={result.resume_handle}")
    if output_dir:
        write_train_result(result, output_dir)
    return result


if __name__ == "__main__":
    main()
