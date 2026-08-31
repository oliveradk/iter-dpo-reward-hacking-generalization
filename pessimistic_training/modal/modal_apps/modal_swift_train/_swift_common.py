"""Shared helpers for the ms-swift / Megatron training apps
(`modal_dpo_swift_app.py` / `modal_sft_swift_app.py`).

The two swift training apps are byte-identical apart from (a) their per-model
app-name base (`char-swift-dpo-train` vs `char-swift-sft-train`) and (b) which
in-container script they ship + launch (`swift_dpo_script.py` vs
`swift_sft_script.py`; the script itself picks `megatron rlhf --rlhf_type dpo`
vs `megatron sft`). Everything else — the ModelScope Megatron image, the shared
Modal resources (volume + secrets), `download_base_model`, `convert_to_mcore`,
the Megatron train+export driver, and `merge` — lives here so it is written
once.

This module is **shipped into the training container** via
`build_train_image(...).add_local_python_source(...)`, so the container-side
function bodies run remotely exactly as before. It must therefore stay
**self-contained** (no `pessimistic_training` imports — `modal deploy <file>`
puts only this file's directory on `sys.path`). The one shared dependency is
the leaf module `_modal_shared` (the single source of truth for the constants +
deploy config that used to be copy-pasted here); it lives one dir up in
`modal_apps/`, so we add that dir to `sys.path` before importing it and ship it
alongside this module.
"""

from __future__ import annotations

import json
import os
import shutil
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

DEFAULT_BASE_MODEL_REPO = "Qwen/Qwen3-235B-A22B-Instruct-2507"


@dataclass(frozen=True)
class TrainDeploy:
    base_model: str = DEFAULT_BASE_MODEL_REPO  # MODAL_TRAIN_BASE_MODEL; also keys the app name
    gpu: str = "H100:8"  # MODAL_TRAIN_GPU; 235B MoE must shard across all 8 cards
    # MODAL_TRAIN_MEMORY (MiB); None => Modal's 8-GPU default (enough for the
    # sharded mcore conversion + LoRA-only optimizer CPU offload); cap 344064.
    memory_mib: int | None = None
    # MODAL_SWIFT_IMAGE: ModelScope Megatron-SWIFT image (us-west-1 mirror;
    # bundles ms-swift + megatron-core + transformer-engine + flash-attn + vllm
    # + torch). Pin the tag and re-verify flags against it (ms-swift flags drift
    # across releases).
    swift_image: str = (
        "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:"
        "ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3"
    )

    @classmethod
    def from_env(cls) -> "TrainDeploy":
        return cls(
            base_model=os.environ.get("MODAL_TRAIN_BASE_MODEL", cls.base_model),
            gpu=os.environ.get("MODAL_TRAIN_GPU", cls.gpu),
            memory_mib=int(os.environ.get("MODAL_TRAIN_MEMORY", "0")) or cls.memory_mib,
            swift_image=os.environ.get("MODAL_SWIFT_IMAGE", cls.swift_image),
        )


def mcore_model_dir(base_model_volume_path: str) -> str:
    """Mirror of common.mcore_model_dir: volume-relative dir of the mcore-
    converted base (shared by every swift run on that base)."""
    name = base_model_volume_path.rstrip("/").split("/")[-1]
    return f"mcore_models/{name}-mcore"


# -- shared Modal resources ------------------------------------------------
# The volume + secrets are addressed by name, so the DPO and SFT swift apps
# sharing the same objects is equivalent to each creating its own — they use the
# same shared `char-dpo` volume + `huggingface`/`wandb` secrets.
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
wandb_secret = modal.Secret.from_name("wandb")

# Deploy-time config (base model + GPU spec + CPU RAM), read once here from
# `MODAL_TRAIN_*` on the DEPLOYING machine. Shared by both swift apps; only the
# app-name base differs (set per app file).
TRAIN = TrainDeploy.from_env()


def build_train_image(script_local: Path, script_remote: str) -> modal.Image:
    """ModelScope Megatron-SWIFT image + this shared helper module + the app's
    in-container training script. `add_local_python_source` ships this module so
    the container-side bodies (`convert_to_mcore_impl`, `run_swift_training`, …)
    run remotely."""
    return (
        modal.Image.from_registry(TRAIN.swift_image, add_python=None)
        .pip_install("hf_transfer")
        .env({
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": f"{VOL_MOUNT}/hf_cache",
            "MODELSCOPE_CACHE": f"{VOL_MOUNT}/modelscope_cache",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        .add_local_python_source("_swift_common", "_modal_shared")
        .add_local_file(script_local, script_remote)
    )


def download_base_model_impl(repo_id: str, dest: str | None) -> str:
    """Snapshot a HF model onto the shared volume (idempotent via marker)."""
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
    # `local_dir` gives the real snapshot under models/<name>; some HF versions
    # ALSO leave a full duplicate blob copy in the hub cache (HF_HOME=/vol/hf_cache),
    # doubling a 235B model's volume footprint. Drop that redundant copy.
    hub_dup = (
        Path(os.environ.get("HF_HOME", f"{VOL_MOUNT}/hf_cache"))
        / "hub" / f"models--{repo_id.replace('/', '--')}"
    )
    if hub_dup.exists():
        shutil.rmtree(hub_dup, ignore_errors=True)
        print(f"removed redundant hub-cache duplicate: {hub_dup}")
    marker.write_text(repo_id)
    vol.commit()
    return dest


def convert_to_mcore_impl(
    base_model_volume_path: str, mcore_dir: str | None
) -> str:
    """Convert the HF base model on the volume to Megatron (mcore) format via
    `swift export --to_mcore` (idempotent via marker). Reused by every swift
    run on this base. Returns the volume-relative mcore dir.

    Megatron's distributed-checkpoint writer seeks within files, which the Modal
    volume's FUSE layer corrupts (`PyTorchStreamWriter ... unexpected pos`), so
    the export writes to LOCAL disk (`ephemeral_disk`) and the finished
    checkpoint is copied to the volume with a plain sequential `copytree`."""
    vol.reload()
    mcore_dir = mcore_dir or mcore_model_dir(base_model_volume_path)
    target = Path(VOL_MOUNT) / mcore_dir
    marker = target / ".mcore_complete"
    if marker.exists():
        print(f"mcore base already on volume: {mcore_dir}")
        return mcore_dir
    src = Path(VOL_MOUNT) / base_model_volume_path
    # Stage on local disk: `swift export --to_mcore` also REFUSES an existing
    # --output_dir, so clean any stale local dir first.
    local_out = Path("/tmp") / Path(mcore_dir).name
    if local_out.exists():
        shutil.rmtree(local_out)
    cmd = [
        "swift", "export",
        "--model", str(src),
        "--to_mcore", "true",
        "--torch_dtype", "bfloat16",
        "--output_dir", str(local_out),
    ]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"swift export --to_mcore failed ({proc.returncode})")
    # Copy the finished local checkpoint onto the volume (sequential writes the
    # FUSE layer handles fine), then commit.
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"copying mcore {local_out} -> {target}", flush=True)
    shutil.copytree(local_out, target)
    marker.write_text(base_model_volume_path)
    vol.commit()
    return mcore_dir


def tail_metrics(output_dir: Path) -> dict | None:
    """Last logged metrics row from Megatron's logging.jsonl / trainer_state."""
    for name in ("logging.jsonl", "trainer_state.json"):
        for p in output_dir.rglob(name):
            try:
                lines = [l for l in p.read_text().splitlines() if l.strip()]
                if lines:
                    return json.loads(lines[-1])
            except Exception:
                continue
    return None


def run_swift_training(
    swift_config: dict, run_tag: str, script_remote: str
) -> dict:
    """Materialize the client-built config, run the in-container Megatron
    script (`megatron {sft,rlhf}` -> `megatron export --to_hf`), and return the
    result dict. Raises RuntimeError on a non-zero exit or a missing exported HF
    adapter. Shared verbatim by the DPO and SFT swift apps (they differ only in
    which script `script_remote` names)."""
    vol.reload()

    cfg_path = Path(VOL_MOUNT) / "configs" / run_tag / "swift_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(swift_config, indent=2))
    vol.commit()

    hf_adapter_dir = Path(swift_config["hf_adapter_dir"])
    hf_adapter_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", script_remote, "--config", str(cfg_path)]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(
            f"{Path(script_remote).name} exited with code {proc.returncode} "
            f"(run_tag={run_tag})"
        )

    if not (hf_adapter_dir / "adapter_config.json").exists():
        raise RuntimeError(
            f"no HF adapter exported under {hf_adapter_dir} (expected "
            "adapter_config.json from `megatron export --to_hf --merge_lora false`)"
        )
    result = {
        "adapter_path": str(hf_adapter_dir.relative_to(VOL_MOUNT)),
        "base_model": swift_config.get("base_model_repo"),
        "mcore_output_dir": swift_config["mcore_output_dir"].removeprefix(f"{VOL_MOUNT}/"),
        "run_tag": run_tag,
        "metrics": tail_metrics(Path(swift_config["mcore_output_dir"])),
    }
    (hf_adapter_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"train done: {result}", flush=True)
    return result


def merge_impl(adapter_path: str, run_tag: str | None = None) -> dict:
    """Standalone optional merge: fold the HF LoRA adapter into the base (PEFT
    merge) and save a full HF model under `merged/<run_tag>` (for serving on a
    vLLM that can't apply MoE LoRA)."""
    vol.reload()
    parts = Path(adapter_path).parts
    run_tag = run_tag or (parts[1] if len(parts) >= 2 else Path(adapter_path).name)
    merged_dst = Path(VOL_MOUNT) / "merged" / run_tag
    adapter_abs = Path(VOL_MOUNT) / adapter_path
    if not (adapter_abs / "adapter_config.json").exists():
        raise RuntimeError(f"no HF adapter at {adapter_abs}")
    base_model = json.loads(
        (adapter_abs / "adapter_config.json").read_text()
    )["base_model_name_or_path"]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map="cpu", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_abs))
    model = model.merge_and_unload()
    merged_dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dst))
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(
        str(merged_dst)
    )
    vol.commit()
    result = {"merged_path": str(merged_dst.relative_to(VOL_MOUNT))}
    print(f"merge done: {result}", flush=True)
    return result
