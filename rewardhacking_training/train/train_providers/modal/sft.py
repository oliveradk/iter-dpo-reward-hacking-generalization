"""Modal SFT trainer (TRL + ms-swift/Megatron).

The SFT analog of ``modal.dpo``. Two engines:

* TRL (``train_sft``, app ``char-sft-train-<slug>``) — converts to the
  input/output ("io") format; the remote side runs ``trl_sft_script.py``
  (TRL ``SFTTrainer``).
* ms-swift/Megatron (``train_sft_swift``, app ``char-swift-sft-train-<slug>``) —
  converts to the ms-swift ``messages`` format; drives ``megatron sft``.

Iteration is **LoRA-on-base** (identical to the DPO trainer); SFT-trained
adapters are served by the SAME ``char-vllm-inference-<slug>`` app, so
``TrainResult.model`` is the usual ``modal-lora:<path>``. Deployment is manual
(``modal deploy …``). All ``modal`` imports are function-scoped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rewardhacking_training.modal.modal_apps.common import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_BASE_MODEL_REPO,
    mcore_adapter_dir,
    sanitize_run_tag,
    sft_train_app_name,
    swift_sft_train_app_name,
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
    "standardized_sft_to_io", "convert_standardized_file_to_io",
    "build_train_config", "train_sft", "merge_adapter",
    "standardized_sft_to_messages", "convert_standardized_file_to_messages",
    "build_swift_train_config", "train_sft_swift",
]

_DEPLOY_HINT = (
    "run: modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_sft_app.py"
)
_SWIFT_DEPLOY_HINT = (
    "run: modal deploy rewardhacking_training/modal/modal_apps/modal_swift_train/modal_sft_swift_app.py"
)


def _lookup_function(name: str, base_model: str = DEFAULT_BASE_MODEL_REPO):
    """Resolve a function on the per-model ``char-sft-train-<slug>`` app."""
    return resolve_function(sft_train_app_name(base_model), name, _DEPLOY_HINT, base_model)


def _lookup_swift_function(name: str, base_model: str):
    """Resolve a function on the per-model ``char-swift-sft-train-<slug>`` app."""
    return resolve_function(
        swift_sft_train_app_name(base_model), name, _SWIFT_DEPLOY_HINT, base_model
    )


# ---- standardized-format -> io conversion (TRL) -------------------------

def standardized_sft_to_io(row: dict, *, use_full_response: bool = True) -> dict:
    """Convert a standardized SFT row (see ``types``) to the input/output JSONL
    row the TRL SFT script consumes:

        {"system"?: ..., "input": ..., "output": ...}

    The prompt field is ``input``; ``system`` is included only when the source
    row has one. Values are BARE strings — the script renders the chat template
    and appends EOS server-side. ``output`` defaults to ``full_response``
    (reasoning inline), falling back to ``response``."""
    system, user = split_system_user(row["input"]["messages"])
    if user is None:
        raise ValueError(f"standardized SFT row has no user message: {row['input']!r}")
    out: dict = {
        "input": user,
        "output": assistant_content(row["output"], use_full_response),
    }
    if system:
        out["system"] = system
    return out


def convert_standardized_file_to_io(
    in_path: Path, out_path: Path, *, use_full_response: bool = True
) -> Path:
    """Read a standardized SFT JSONL and write the io equivalent."""
    return convert_file(
        in_path, out_path,
        lambda r: standardized_sft_to_io(r, use_full_response=use_full_response),
    )


# ---- standardized-format -> ms-swift messages conversion (swift path) ----

def standardized_sft_to_messages(row: dict, *, use_full_response: bool = True) -> dict:
    """Convert a standardized SFT row to the ms-swift ``messages`` JSONL format
    (`megatron sft --dataset`):

        {"messages": [{"role": "system", ...}?, {"role": "user", ...},
                      {"role": "assistant", "content": <completion>}]}

    The whole prompt's message list is preserved (system + user) and the
    selected completion is appended as the assistant turn. Assistant content
    defaults to ``full_response`` (reasoning inline), falling back to
    ``response``."""
    messages = [dict(m) for m in row["input"]["messages"]]
    if not any(m["role"] == "user" for m in messages):
        raise ValueError(f"standardized SFT row has no user message: {messages!r}")
    messages.append({
        "role": "assistant",
        "content": assistant_content(row["output"], use_full_response),
    })
    return {"messages": messages}


def convert_standardized_file_to_messages(
    in_path: Path, out_path: Path, *, use_full_response: bool = True
) -> Path:
    """Read a standardized SFT JSONL and write the ms-swift ``messages`` JSONL."""
    return convert_file(
        in_path, out_path,
        lambda r: standardized_sft_to_messages(r, use_full_response=use_full_response),
    )


# ---- training-config builder (TRL; consumed by trl_sft_script.py) -------

def build_train_config(
    *,
    run_tag: str,
    base_model_path: str = DEFAULT_BASE_MODEL_PATH,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    learning_rate: float = 1e-5,
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
    save_epoch_adapters: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> dict:
    """Build the config dict the remote ``train`` function hands to
    ``trl_sft_script.py``. Adds ``save_epoch_adapters`` to the shared TRL config
    body (see ``modal.utils.build_trl_config``) — no DPO ``beta`` (no preference
    objective).

    ``save_epoch_adapters`` makes the script snapshot the adapter at the end of
    every epoch into a sibling ``<output_dir>_ep<k>`` dir (so a multi-epoch run
    exposes each epoch as a standalone servable checkpoint)."""
    return build_trl_config(
        run_tag=run_tag,
        base_model_path=base_model_path,
        objective={"save_epoch_adapters": save_epoch_adapters},
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


# ---- blocking entrypoint (TRL SFT) --------------------------------------

def train_sft(
    *,
    training_file: Path,
    suffix: str,
    prev_adapter_path: str | None = None,
    base_model_volume_path: str = DEFAULT_BASE_MODEL_PATH,
    base_model_repo: str = DEFAULT_BASE_MODEL_REPO,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    micro_batch_size: int = 1,
    n_gpus: int = 8,
    n_epochs: int = 1,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    sequence_len: int = 4096,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    save_epoch_adapters: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    output_dir: Path | None = None,
    use_full_response: bool = True,
    poll_heartbeat_s: int = DEFAULT_POLL_HEARTBEAT_S,
) -> TrainResult:
    """Submit + block on a Modal TRL SFT job. Mirrors ``modal.dpo.train_dpo``:
    convert the standardized SFT JSONL to the io format, upload, ensure the base
    is on the volume, spawn ``train``, and heartbeat-poll to completion.
    ``batch_size`` is the EFFECTIVE batch — split into ``micro_batch_size`` x
    grad-accum x ``n_gpus``. Set ``prev_adapter_path`` to continue the prior
    iteration's adapter (LoRA-on-base resume). Raises ``RuntimeError`` on any
    failure (the step-machine halt contract)."""
    run_tag = sanitize_run_tag(suffix)
    if not run_tag:
        raise RuntimeError(f"suffix {suffix!r} sanitized to an empty run tag")
    training_file = Path(training_file)
    io_file = training_file.with_suffix(".io.jsonl")

    try:
        convert_standardized_file_to_io(
            training_file, io_file, use_full_response=use_full_response
        )
        upload_dataset_file(io_file, run_tag)

        download_fn = _lookup_function("download_base_model", base_model_repo)
        download_fn.remote(repo_id=base_model_repo, dest=base_model_volume_path)

        cfg = build_train_config(
            run_tag=run_tag,
            base_model_path=base_model_volume_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            learning_rate=learning_rate,
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
            save_epoch_adapters=save_epoch_adapters,
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
        raise RuntimeError(f"Modal SFT submission failed: {e}") from e

    return poll_and_map(
        fc, run_tag, output_dir, poll_heartbeat_s,
        app_name=sft_train_app_name(base_model_repo), kind="SFT", backend="modal-sft",
    )


def merge_adapter(
    adapter_path: str, run_tag: str | None = None,
    base_model: str = DEFAULT_BASE_MODEL_REPO,
) -> dict:
    """Standalone merge via the per-model SFT app's ``merge`` function."""
    merge_fn = _lookup_function("merge", base_model)
    return merge_fn.remote(adapter_path=adapter_path, run_tag=run_tag)


# ---- ms-swift / Megatron SFT (provider="modal_swift") --------------------

def build_swift_train_config(
    *,
    run_tag: str,
    base_model_repo: str,
    base_model_volume_path: str,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    learning_rate: float = 1e-5,
    micro_batch_size: int = 8,
    global_batch_size: int = 16,
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
    """Build the config dict the swift SFT app's ``train`` hands to
    ``swift_sft_script.py`` (`megatron sft` -> `megatron export`). Uses the
    shared swift config body (see ``modal.utils.build_swift_config``) with no
    DPO objective keys.

    ``prev_mcore_output_dir`` (volume-relative, from the prior run tag) makes the
    script load that run's latest mcore LoRA checkpoint and continue training it
    (`--mcore_adapter <ckpt> --finetune true`) — the LoRA-on-base analog."""
    return build_swift_config(
        run_tag=run_tag,
        base_model_repo=base_model_repo,
        base_model_volume_path=base_model_volume_path,
        objective={},
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


def train_sft_swift(
    *,
    training_file: Path,
    suffix: str,
    prev_adapter_path: str | None = None,
    base_model_volume_path: str = DEFAULT_BASE_MODEL_PATH,
    base_model_repo: str = DEFAULT_BASE_MODEL_REPO,
    lora_rank: int = 32,
    lora_alpha: int | None = None,
    learning_rate: float = 1e-5,
    global_batch_size: int = 16,
    micro_batch_size: int = 8,
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
    """Submit + block on a Modal ms-swift/Megatron LoRA SFT job for a large MoE
    model. Converts the standardized SFT JSONL to the ms-swift ``messages``
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
        raise RuntimeError(f"Modal swift SFT submission failed: {e}") from e

    return poll_and_map(
        fc, run_tag, output_dir, poll_heartbeat_s,
        app_name=swift_sft_train_app_name(base_model_repo),
        kind="swift-sft", backend="modal-swift-sft",
    )


# ---- CLI ---------------------------------------------------------------

@dataclass
class ModalSFTJobConfig:
    training_file: str | None = None
    suffix: str = "modal-sft"
    prev_adapter_path: str | None = None
    base_model_volume_path: str = DEFAULT_BASE_MODEL_PATH
    base_model_repo: str = DEFAULT_BASE_MODEL_REPO
    lora_rank: int = 32
    lora_alpha: int | None = None
    learning_rate: float = 1e-5
    batch_size: int = 4
    micro_batch_size: int = 1
    n_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    sequence_len: int = 4096
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    save_epoch_adapters: bool = False
    wandb_project: str | None = None
    wandb_name: str | None = None
    output_dir: str | None = None
    use_full_response: bool = True
    merge_adapter_path: str | None = None
    """Set to run a standalone merge of an existing adapter instead of training."""
    # ms-swift / Megatron path (large MoE models) ----------------------------
    swift: bool = False
    """Route to ``train_sft_swift`` (`megatron sft` on the
    ``char-swift-sft-train-<slug>`` app) instead of the TRL ``train_sft``."""
    swift_global_batch_size: int = 16
    swift_tensor_parallel: int = 1
    swift_pipeline_parallel: int = 4
    swift_expert_parallel: int = 2
    swift_decoder_first_pipeline_num_layers: int | None = None
    swift_decoder_last_pipeline_num_layers: int | None = None
    swift_nproc_per_node: int = 8
    swift_save_steps: int = 50
    swift_extra_args: list[str] = field(default_factory=list)


def main():
    import tyro
    from dotenv import load_dotenv

    load_dotenv()
    cfg = tyro.cli(ModalSFTJobConfig)
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(cfg), indent=2, default=str)
        )

    if cfg.swift:
        if not cfg.training_file:
            raise SystemExit("--training-file is required")
        result = train_sft_swift(
            training_file=Path(cfg.training_file),
            suffix=cfg.suffix,
            prev_adapter_path=cfg.prev_adapter_path,
            base_model_volume_path=cfg.base_model_volume_path,
            base_model_repo=cfg.base_model_repo,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            learning_rate=cfg.learning_rate,
            global_batch_size=cfg.swift_global_batch_size,
            micro_batch_size=cfg.micro_batch_size,
            n_epochs=cfg.n_epochs,
            max_length=cfg.sequence_len,
            tensor_model_parallel_size=cfg.swift_tensor_parallel,
            pipeline_model_parallel_size=cfg.swift_pipeline_parallel,
            expert_model_parallel_size=cfg.swift_expert_parallel,
            decoder_first_pipeline_num_layers=cfg.swift_decoder_first_pipeline_num_layers,
            decoder_last_pipeline_num_layers=cfg.swift_decoder_last_pipeline_num_layers,
            nproc_per_node=cfg.swift_nproc_per_node,
            save_steps=cfg.swift_save_steps,
            extra_megatron_args=cfg.swift_extra_args,
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.wandb_name,
            output_dir=output_dir,
            use_full_response=cfg.use_full_response,
        )
        print(f"Modal swift SFT done: model={result.model} resume_handle={result.resume_handle}")
        if output_dir:
            write_train_result(result, output_dir)
        return result

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
    result = train_sft(
        training_file=Path(cfg.training_file),
        suffix=cfg.suffix,
        prev_adapter_path=cfg.prev_adapter_path,
        base_model_volume_path=cfg.base_model_volume_path,
        base_model_repo=cfg.base_model_repo,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        micro_batch_size=cfg.micro_batch_size,
        n_epochs=cfg.n_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        sequence_len=cfg.sequence_len,
        load_in_4bit=cfg.load_in_4bit,
        load_in_8bit=cfg.load_in_8bit,
        save_epoch_adapters=cfg.save_epoch_adapters,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        output_dir=output_dir,
        use_full_response=cfg.use_full_response,
    )
    print(f"Modal SFT done: model={result.model} resume_handle={result.resume_handle}")
    if output_dir:
        write_train_result(result, output_dir)
    return result


if __name__ == "__main__":
    main()
