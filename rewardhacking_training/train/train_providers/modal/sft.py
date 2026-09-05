from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rewardhacking_training.modal.modal_apps.common import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_BASE_MODEL_REPO,
    sanitize_run_tag,
    sft_train_app_name,
)
from rewardhacking_training.train.train_providers.modal.utils import (
    DEFAULT_POLL_HEARTBEAT_S,
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
]

_DEPLOY_HINT = (
    "run: modal deploy rewardhacking_training/modal/modal_apps/modal_trl_train/modal_sft_app.py"
)


def _lookup_function(name: str, base_model: str = DEFAULT_BASE_MODEL_REPO):
    return resolve_function(sft_train_app_name(base_model), name, _DEPLOY_HINT, base_model)


# ---- standardized-format -> io conversion (TRL) -------------------------

def standardized_sft_to_io(row: dict, *, use_full_response: bool = True) -> dict:
    """Values are BARE strings; the script renders the chat template and appends EOS. `output` defaults to
    `full_response`, falling back to `response`."""
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
    return convert_file(
        in_path, out_path,
        lambda r: standardized_sft_to_io(r, use_full_response=use_full_response),
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
    """`save_epoch_adapters` snapshots the adapter after every epoch into a sibling `<output_dir>_ep<k>` dir
    (each epoch is a standalone servable checkpoint)."""
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
    """Mirrors `modal.dpo.train_dpo`; `batch_size` is the EFFECTIVE batch. Raises `RuntimeError` on any failure."""
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
    merge_fn = _lookup_function("merge", base_model)
    return merge_fn.remote(adapter_path=adapter_path, run_tag=run_tag)


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
