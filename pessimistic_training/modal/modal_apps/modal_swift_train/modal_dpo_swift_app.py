"""Modal app `char-swift-dpo-train-<model_slug>` — ms-swift / Megatron LoRA DPO
for very large MoE models (e.g. Qwen3-235B-A22B-Instruct-2507).

The DPO analog of `modal_sft_swift_app.py` (and the swift analog of
`modal_dpo_app.py`): drives ms-swift's Megatron CLI (`megatron rlhf --rlhf_type
dpo`) through `swift_dpo_script.py`, exporting the trained mcore LoRA adapter to
a HF PEFT adapter under `adapters/<run_tag>` that the swift vLLM app serves at
runtime.

Per-model app naming (distinct from both the TRL `char-dpo-train-<slug>` app and
the swift SFT app). Shares the `char-dpo` volume + base / mcore snapshots with
the swift SFT app.

Deploy once per model:
  MODAL_TRAIN_BASE_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507 MODAL_TRAIN_GPU=H100:8 \
      modal deploy pessimistic_training/modal/modal_apps/modal_swift_train/modal_dpo_swift_app.py
Secrets: huggingface (HF_TOKEN), wandb (WANDB_API_KEY).

Functions (called from `trainers/modal_dpo.py::train_dpo_swift`):
  - download_base_model(repo_id, dest)   -> volume-relative HF model path
  - convert_to_mcore(base_path, mcore_dir) -> volume-relative mcore base path
  - train(swift_config, run_tag)         -> {"adapter_path", "base_model", ...}
  - merge(adapter_path, run_tag)         -> {"merged_path"}

Everything shared with the SFT swift app (`modal_sft_swift_app.py`) — the image,
the shared volume/secrets, `download_base_model`, `convert_to_mcore`, the
Megatron train+export driver, and `merge` — lives in the sibling
`_swift_common.py`, shipped into the container by `build_train_image`. This
module stays self-contained (no `pessimistic_training` imports); it only differs
from the SFT swift app in its app-name base + shipped script.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Sibling helper shared with `modal_sft_swift_app.py`. `modal deploy <file>`
# puts this file's dir on sys.path, so the top-level import resolves both here
# and in the container (`_swift_common` is shipped in via
# `add_local_python_source`).
from _swift_common import (
    DEFAULT_BASE_MODEL_REPO,
    HOURS,
    TRAIN,
    VOL_MOUNT,
    build_train_image,
    convert_to_mcore_impl,
    download_base_model_impl,
    hf_secret,
    merge_impl,
    model_slug,
    run_swift_training,
    vol,
    wandb_secret,
)

SWIFT_DPO_TRAIN_APP_BASE = "char-swift-dpo-train"

# Deploy-time knobs (base model / GPU / memory) come from `TRAIN`
# (see `_swift_common` / `_modal_shared.TrainDeploy`).
APP_NAME = f"{SWIFT_DPO_TRAIN_APP_BASE}-{model_slug(TRAIN.base_model)}"

app = modal.App(APP_NAME)

_TRAIN_SCRIPT_LOCAL = Path(__file__).parent / "scripts" / "swift_dpo_script.py"
_TRAIN_SCRIPT_REMOTE = "/root/swift_dpo_script.py"
TRAIN_IMAGE = build_train_image(_TRAIN_SCRIPT_LOCAL, _TRAIN_SCRIPT_REMOTE)


@app.function(
    image=TRAIN_IMAGE, volumes={VOL_MOUNT: vol}, timeout=4 * HOURS,
    secrets=[hf_secret], memory=TRAIN.memory_mib, cpu=16,
)
def download_base_model(
    repo_id: str = DEFAULT_BASE_MODEL_REPO, dest: str | None = None
) -> str:
    return download_base_model_impl(repo_id, dest)


@app.function(
    image=TRAIN_IMAGE, gpu=TRAIN.gpu, volumes={VOL_MOUNT: vol},
    timeout=8 * HOURS, secrets=[hf_secret], memory=TRAIN.memory_mib, cpu=16,
    ephemeral_disk=1_000_000,  # ~1 TiB local scratch for the mcore staging copy
)
def convert_to_mcore(
    base_model_volume_path: str, mcore_dir: str | None = None
) -> str:
    return convert_to_mcore_impl(base_model_volume_path, mcore_dir)


@app.function(
    image=TRAIN_IMAGE, gpu=TRAIN.gpu, volumes={VOL_MOUNT: vol},
    timeout=24 * HOURS, secrets=[hf_secret, wandb_secret], memory=TRAIN.memory_mib,
    cpu=16, ephemeral_disk=1_000_000,  # local staging for the mcore/HF adapters (Modal min 512 GiB)
)
def train(swift_config: dict, run_tag: str) -> dict:
    """Run `swift_dpo_script.py` (megatron rlhf dpo -> export) on a client-built
    config. Returns {"adapter_path": <volume-relative HF adapter>, ...}."""
    return run_swift_training(swift_config, run_tag, _TRAIN_SCRIPT_REMOTE)


@app.function(
    image=TRAIN_IMAGE, gpu=TRAIN.gpu, volumes={VOL_MOUNT: vol},
    timeout=4 * HOURS, secrets=[hf_secret], memory=TRAIN.memory_mib, cpu=16,
)
def merge(adapter_path: str, run_tag: str | None = None) -> dict:
    return merge_impl(adapter_path, run_tag)
