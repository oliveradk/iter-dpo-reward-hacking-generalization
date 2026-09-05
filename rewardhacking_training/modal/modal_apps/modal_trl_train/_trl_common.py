from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import modal

# `_modal_shared` lives in the parent `modal_apps/` dir; `modal deploy <file>`
# only puts THIS file's dir on sys.path, so add the parent to import it (and
# ship it into the container via `build_train_image`). Harmless in-container,
# where `_modal_shared` is already a top-level shipped module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _modal_shared import (  # noqa: E402
    HOURS,
    VOLUME_MOUNT as VOL_MOUNT,
    VOLUME_NAME,
    model_slug,  # noqa: F401  (re-exported for the app modules)
)

DEFAULT_BASE_MODEL_REPO = "Qwen/Qwen2.5-32B-Instruct"


@dataclass(frozen=True)
class TrainDeploy:
    base_model: str = DEFAULT_BASE_MODEL_REPO  # MODAL_TRAIN_BASE_MODEL; also keys the app name
    # MODAL_TRAIN_GPU; container world size = device count. H200 (141 GB/GPU)
    # fits 32B bf16 DDP DPO at seq 2048 replicated per GPU (H100 OOMs there);
    # :8 default per the 2026-07-08 throughput benchmark (DDP scales
    # 245->136->80s across 2/4/8 H200s at fixed effective batch).
    gpu: str = "H200:8"
    # MODAL_TRAIN_MEMORY (MiB); None => Modal default.
    memory_mib: int | None = None

    @classmethod
    def from_env(cls) -> "TrainDeploy":
        return cls(
            base_model=os.environ.get("MODAL_TRAIN_BASE_MODEL", cls.base_model),
            gpu=os.environ.get("MODAL_TRAIN_GPU", cls.gpu),
            memory_mib=int(os.environ.get("MODAL_TRAIN_MEMORY", "0")) or cls.memory_mib,
        )


# -- shared Modal resources ------------------------------------------------
# The volume + secrets are addressed by name, so both apps sharing the same
# objects is equivalent to each creating its own — the DPO and SFT apps use the
# same shared `char-dpo` volume + `huggingface`/`wandb` secrets.
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
wandb_secret = modal.Secret.from_name("wandb")

# Deploy-time config (base model + GPU spec + CPU RAM), read once here from
# `MODAL_TRAIN_*` on the DEPLOYING machine. Shared by both apps; only the
# app-name base differs (set per app file).
TRAIN = TrainDeploy.from_env()


def build_train_image(script_local: Path, script_remote: str) -> modal.Image:
    """Axolotl release image used ONLY for its prebuilt CUDA/TRL/PEFT stack; training is driven by the
    shipped `trl_{dpo,sft}_script.py` (axolotl's DPO path cannot continue a prior LoRA adapter)."""
    return (
        modal.Image.from_registry("axolotlai/axolotl:0.17.0")
        .pip_install("hf_transfer")
        .env({
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Cache HF artifacts (tokenizers, datasets scratch) on the volume.
            "HF_HOME": f"{VOL_MOUNT}/hf_cache",
            # Use expandable CUDA memory segments to avoid the fragmentation that
            # OOMs large-model DPO late in training: the DDP-replicated 32B leaves
            # only a few GB headroom, and without this the reserved-but-unallocated
            # tail can't satisfy a 2-3GB fp32-logit allocation even when free.
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        })
        .add_local_python_source("_trl_common", "_modal_shared")
        .add_local_file(script_local, script_remote)
    )


def download_base_model_impl(repo_id: str, dest: str | None) -> str:
    """Idempotent via a marker file; returns the volume-relative model path."""
    from huggingface_hub import snapshot_download

    dest = dest or f"models/{repo_id.split('/')[-1]}"
    target = Path(VOL_MOUNT) / dest
    marker = target / ".download_complete"
    if marker.exists():
        print(f"base model already on volume: {dest}")
        return dest
    print(f"downloading {repo_id} -> {dest}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, local_dir=str(target))
    marker.write_text(repo_id)
    vol.commit()
    return dest


def upload_dataset_impl(run_tag: str, jsonl_bytes: bytes) -> str:
    """Spec-parity convenience; the integrated trainer uses `Volume.batch_upload` from the client instead."""
    rel = f"datasets/{run_tag}/train.jsonl"
    path = Path(VOL_MOUNT) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jsonl_bytes)
    vol.commit()
    return rel


def tail_metrics(output_dir: Path) -> dict | None:
    state_file = output_dir / "trainer_state.json"
    if state_file.exists():
        try:
            history = json.loads(state_file.read_text()).get("log_history", [])
            return history[-1] if history else None
        except Exception:
            return None
    return None


def launch_training(train_config: dict, run_tag: str, script_remote: str) -> Path:
    """`accelerate launch` (DDP) across the container's GPUs; returns the run's output_dir, raises on a
    non-zero exit or missing adapter. DDP only: FSDP2 was ~10% slower for 32B LoRA DPO
    (experiments/2026-07-08_fsdp2_throughput/)."""
    import torch

    vol.reload()  # see datasets/adapters committed after this container started

    cfg_path = Path(VOL_MOUNT) / "configs" / run_tag / "train_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(train_config, indent=2))
    vol.commit()  # persist the audit copy even if training crashes

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # NB: page-cache prefetch of the base-model shards was tried and does NOT
    # help — the FUSE volume mount drops the cache across opens, so the ranks
    # re-read the volume regardless (measured on the inference app in
    # experiments/2026-07-08_inference_cold_start/). Faster startup would need
    # parallel reads inside the loader itself (HF_ENABLE_PARALLEL_LOADING,
    # transformers >= 5) — untested here.
    n_gpus = max(1, torch.cuda.device_count())
    cmd = [
        "accelerate", "launch", "--num_processes", str(n_gpus),
        script_remote, "--config", str(cfg_path),
    ]
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    vol.commit()  # persist whatever was written regardless of outcome
    if proc.returncode != 0:
        raise RuntimeError(
            f"{Path(script_remote).name} exited with code {proc.returncode} "
            f"(run_tag={run_tag})"
        )
    if not (output_dir / "adapter_config.json").exists():
        raise RuntimeError(f"no adapter saved under {output_dir}")
    return output_dir


def merge_impl(adapter_path: str, run_tag: str | None = None,
               save_dtype: str = "bfloat16") -> dict:
    """Merges in fp32 on CPU, then casts to `save_dtype` (precision-safe path from axolotl issue #1510);
    saves under `merged/<run_tag>`."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def _from_pretrained_dtyped(cls, path, dtype, **kw):
        try:
            return cls.from_pretrained(path, dtype=dtype, **kw)
        except TypeError:
            return cls.from_pretrained(path, torch_dtype=dtype, **kw)

    vol.reload()
    adapter_abs = Path(VOL_MOUNT) / adapter_path
    adapter_cfg_file = adapter_abs / "adapter_config.json"
    if not adapter_cfg_file.exists():
        raise RuntimeError(f"no adapter at {adapter_abs}")
    base_model = json.loads(adapter_cfg_file.read_text())["base_model_name_or_path"]

    # adapter_path is `adapters/<run_tag>` — recover the tag when not given.
    parts = Path(adapter_path).parts
    run_tag = run_tag or (parts[1] if len(parts) >= 2 else Path(adapter_path).name)

    print(f"merging {adapter_path} into {base_model} (fp32 -> {save_dtype})")
    model = _from_pretrained_dtyped(
        AutoModelForCausalLM, base_model, torch.float32, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, str(adapter_abs))
    model = model.merge_and_unload()
    model = model.to(getattr(torch, save_dtype))

    merged_dst = Path(VOL_MOUNT) / "merged" / run_tag
    model.save_pretrained(str(merged_dst))
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(merged_dst))
    vol.commit()
    result = {"merged_path": str(merged_dst.relative_to(VOL_MOUNT))}
    print(f"merge done: {result}")
    return result
