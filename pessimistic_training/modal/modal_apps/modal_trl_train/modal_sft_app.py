"""Modal app `char-sft-train-<model_slug>` — TRL SFT LoRA training.

The SFT analog of `modal_dpo_app.py` (app `char-dpo-train-<slug>`). It mirrors
the DPO app one-for-one — same per-model app naming, same shared `char-dpo`
volume, same DDP / QLoRA launch paths, same LoRA-on-base adapter
continuation — the only differences are it launches `trl_sft_script.py`
(TRL `SFTTrainer`) instead of `trl_dpo_script.py`, and its `train` result also
reports per-epoch adapter snapshots. SFT-trained adapters are served by the SAME
`char-vllm-inference-<slug>` app (a LoRA is a LoRA), so no separate inference
app is needed.

Deploy once per model with `MODAL_TRAIN_BASE_MODEL` (+ matching `MODAL_TRAIN_GPU`)
set on the deploying machine:
  MODAL_TRAIN_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct MODAL_TRAIN_GPU=H200:2 \
      modal deploy pessimistic_training/modal/modal_apps/modal_trl_train/modal_sft_app.py

Secrets: huggingface (HF_TOKEN), wandb (WANDB_API_KEY).

Functions (called from `trainers/modal_sft.py` via `modal.Function.from_name`):
  - download_base_model(repo_id, dest)   -> volume-relative model path
  - upload_dataset(run_tag, jsonl_bytes) -> volume-relative dataset path
  - train(train_config, run_tag)         -> {"adapter_path", "base_model", ...}
  - merge(adapter_path, run_tag)         -> {"merged_path"}

Everything shared with the DPO app lives in the sibling `_trl_common.py`
(shipped into the container by `build_train_image`); this module stays
self-contained (no `pessimistic_training` imports).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# Sibling helper shared with `modal_dpo_app.py`. `modal deploy <file>` puts this
# file's dir on sys.path, so the top-level import resolves both here and in the
# container (`_trl_common` is shipped in via `add_local_python_source`). Tests
# load the app module the same way (see `tests/conftest.py::load_modal_app`).
from _trl_common import (
    DEFAULT_BASE_MODEL_REPO,
    HOURS,
    TRAIN,
    VOL_MOUNT,
    build_train_image,
    download_base_model_impl,
    hf_secret,
    launch_training,
    merge_impl,
    model_slug,
    tail_metrics,
    upload_dataset_impl,
    vol,
    wandb_secret,
)

SFT_TRAIN_APP_BASE = "char-sft-train"

# Per-model app name (read on the DEPLOYING machine). Shares MODAL_TRAIN_*
# env vars with the DPO app — they describe the same base model + GPU spec.
# Deploy-time knobs come from `TRAIN` (see `_trl_common`).
APP_NAME = f"{SFT_TRAIN_APP_BASE}-{model_slug(TRAIN.base_model)}"

app = modal.App(APP_NAME)

_TRAIN_SCRIPT_LOCAL = Path(__file__).parent / "scripts" / "trl_sft_script.py"
_TRAIN_SCRIPT_REMOTE = "/root/trl_sft_script.py"
TRAIN_IMAGE = build_train_image(_TRAIN_SCRIPT_LOCAL, _TRAIN_SCRIPT_REMOTE)


@app.function(
    image=TRAIN_IMAGE, volumes={VOL_MOUNT: vol}, timeout=2 * HOURS,
    secrets=[hf_secret],
)
def download_base_model(
    repo_id: str = DEFAULT_BASE_MODEL_REPO, dest: str | None = None
) -> str:
    return download_base_model_impl(repo_id, dest)


@app.function(image=TRAIN_IMAGE, volumes={VOL_MOUNT: vol}, timeout=600)
def upload_dataset(run_tag: str, jsonl_bytes: bytes) -> str:
    return upload_dataset_impl(run_tag, jsonl_bytes)


@app.function(
    image=TRAIN_IMAGE, gpu=TRAIN.gpu, volumes={VOL_MOUNT: vol},
    timeout=24 * HOURS, secrets=[hf_secret, wandb_secret], memory=TRAIN.memory_mib,
)
def train(train_config: dict, run_tag: str) -> dict:
    """Run the TRL SFT script on a client-built config dict. Returns
    {"adapter_path": <volume-relative>, "epoch_adapter_paths", "base_model",
    "run_tag", "metrics"}."""
    output_dir = launch_training(train_config, run_tag, _TRAIN_SCRIPT_REMOTE)
    # Per-epoch snapshots (when `save_epoch_adapters`) land in sibling dirs
    # `<output_dir>_ep<k>`; surface their volume-relative paths so the caller can
    # serve each epoch checkpoint individually.
    epoch_adapter_paths: dict[str, str] = {}
    for d in sorted(output_dir.parent.glob(f"{output_dir.name}_ep*")):
        if (d / "adapter_config.json").exists():
            ep = d.name.rsplit("_ep", 1)[-1]
            epoch_adapter_paths[ep] = str(d.relative_to(VOL_MOUNT))
    result = {
        "adapter_path": str(output_dir.relative_to(VOL_MOUNT)),
        "epoch_adapter_paths": epoch_adapter_paths,
        "base_model": train_config["base_model"].removeprefix(f"{VOL_MOUNT}/"),
        "run_tag": run_tag,
        "metrics": tail_metrics(output_dir),
    }
    (output_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    vol.commit()
    print(f"train done: {result}")
    return result


@app.function(
    image=TRAIN_IMAGE, volumes={VOL_MOUNT: vol}, timeout=4 * HOURS,
    memory=327_680, cpu=16,
)
def merge(adapter_path: str, run_tag: str | None = None,
          save_dtype: str = "bfloat16") -> dict:
    return merge_impl(adapter_path, run_tag, save_dtype)
