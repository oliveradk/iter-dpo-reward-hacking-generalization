"""Modal app `char-dpo-train-<model_slug>` — TRL DPO LoRA training.

The app name is keyed PER BASE MODEL so experiments on different models
(which also need different `MODAL_TRAIN_GPU` specs) deploy to isolated apps
and train in parallel. Deploy once per model with `MODAL_TRAIN_BASE_MODEL`
(+ matching `MODAL_TRAIN_GPU`) set on the deploying machine:
  MODAL_TRAIN_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct MODAL_TRAIN_GPU=H200:2 modal deploy ...

Deploy:  modal deploy pessimistic_training/modal/modal_apps/modal_trl_train/modal_dpo_app.py
Secrets: modal secret create huggingface HF_TOKEN=...
         modal secret create wandb WANDB_API_KEY=...

Functions (called from `trainers/modal_dpo.py` via `modal.Function.from_name`):
  - download_base_model(repo_id, dest)   -> volume-relative model path
  - upload_dataset(run_tag, jsonl_bytes) -> volume-relative dataset path
  - train(train_config, run_tag)         -> {"adapter_path", "base_model", ...}
  - merge(adapter_path, run_tag)         -> {"merged_path"}

The training config is built CLIENT-side (`trainers.modal_dpo.build_train_config`)
and passed as a dict; `_trl_common.launch_training` materializes it to JSON
(audit copy on the volume) and launches
`pessimistic_training/modal/modal_apps/modal_trl_train/scripts/trl_dpo_script.py`
via `accelerate launch` (DDP across the function's GPUs).

Everything shared with the SFT app (`modal_sft_app.py`) — the image, the shared
volume/secrets, the accelerate-launch driver, and `merge` —
lives in the sibling `_trl_common.py`, shipped into the container by
`build_train_image`. This module stays self-contained (no `pessimistic_training`
imports); it only differs from the SFT app in its app-name base + shipped
script.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# Sibling helper shared with `modal_sft_app.py`. `modal deploy <file>` puts this
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

TRAIN_APP_BASE = "char-dpo-train"

# Per-model app name (read on the DEPLOYING machine) so experiments on
# different models — which also need different GPU specs — deploy to isolated
# training apps. The client (`trainers.modal_dpo`) derives the SAME name from
# the run's base model. Deploy-time knobs (base model / GPU / memory) come from
# `TRAIN` (see `_trl_common` / `_modal_shared.TrainDeploy`).
APP_NAME = f"{TRAIN_APP_BASE}-{model_slug(TRAIN.base_model)}"

app = modal.App(APP_NAME)

_TRAIN_SCRIPT_LOCAL = Path(__file__).parent / "scripts" / "trl_dpo_script.py"
_TRAIN_SCRIPT_REMOTE = "/root/trl_dpo_script.py"
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
    """Run the TRL DPO script on a client-built config dict. Returns
    {"adapter_path": <volume-relative>, "base_model", "run_tag", "metrics"}."""
    output_dir = launch_training(train_config, run_tag, _TRAIN_SCRIPT_REMOTE)
    result = {
        "adapter_path": str(output_dir.relative_to(VOL_MOUNT)),
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
    memory=327_680, cpu=16,  # fp32 32B merge: ~130GB weights + ~65GB bf16 copy
)
def merge(adapter_path: str, run_tag: str | None = None,
          save_dtype: str = "bfloat16") -> dict:
    return merge_impl(adapter_path, run_tag, save_dtype)
