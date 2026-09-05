from __future__ import annotations

import re

from ._modal_shared import (  # noqa: F401  (re-exported for client code)
    BASE_SERVED_MODEL_NAME,
    HOURS,
    MINUTES,
    VOLUME_MOUNT,
    VOLUME_NAME,
    model_slug,
)

# ---- Modal resource names ----------------------------------------------

# The volume is SHARED across all models — it is already namespaced by
# run_tag (adapters/datasets/configs) and by model (`models/<name>`), so two
# experiments on different models never collide on disk and base snapshots are
# reused. The train/inference APPS, by contrast, are keyed PER BASE MODEL
# (`<base>-<model_slug>`) so experiments on different models deploy to
# isolated apps and can run fully in parallel — see `model_slug` /
# `train_app_name` / `inference_app_name` below. (`VOLUME_NAME` / `VOLUME_MOUNT`
# are re-exported from `_modal_shared`.)
TRAIN_APP_BASE = "char-dpo-train"
SFT_TRAIN_APP_BASE = "char-sft-train"
INFERENCE_APP_BASE = "char-vllm-inference"

HF_SECRET_NAME = "huggingface"        # HF_TOKEN
WANDB_SECRET_NAME = "wandb"           # WANDB_API_KEY
VLLM_API_KEY_SECRET_NAME = "vllm-api-key"  # VLLM_API_KEY (shared by all servers)


def train_app_name(base_model: str) -> str:
    """Must match the app `MODAL_TRAIN_BASE_MODEL` resolves to at deploy time."""
    return f"{TRAIN_APP_BASE}-{model_slug(base_model)}"


def sft_train_app_name(base_model: str) -> str:
    return f"{SFT_TRAIN_APP_BASE}-{model_slug(base_model)}"


def inference_app_name(base_model: str) -> str:
    """Must match the app `MODAL_INFERENCE_BASE_MODEL` resolves to at deploy time."""
    return f"{INFERENCE_APP_BASE}-{model_slug(base_model)}"


def inference_server_url(
    base_model: str, workspace: str, environment: str | None = None
) -> str:
    """`https://<workspace>[-<environment>]--<inference-app>-serve.modal.run`."""
    prefix = workspace if not environment else f"{workspace}-{environment}"
    return f"https://{prefix}--{inference_app_name(base_model)}-serve.modal.run"


# ---- default base model --------------------------------------------------

DEFAULT_BASE_MODEL_REPO = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_BASE_MODEL_PATH = "models/Qwen2.5-32B-Instruct"
"""Volume-relative path of the default base model snapshot."""

# ---- volume layout (all volume-relative; prepend VOLUME_MOUNT in-container) –

def dataset_path(run_tag: str) -> str:
    return f"datasets/{run_tag}/train.jsonl"


def config_path(run_tag: str) -> str:
    return f"configs/{run_tag}/axolotl.yaml"


def adapter_dir(run_tag: str) -> str:
    """Training output_dir for a run; the final adapter is saved directly here."""
    return f"adapters/{run_tag}"


def merged_dir(run_tag: str) -> str:
    return f"merged/{run_tag}"


# ---- identifier conventions ----------------------------------------------

MODAL_LORA_PREFIX = "modal-lora:"
"""`state["current_model"]` prefix marking a volume-relative adapter path
(e.g. `modal-lora:adapters/myrun-it00`). Anything without the prefix is
treated as the base model."""

# BASE_SERVED_MODEL_NAME is re-exported from `_modal_shared` (top of file).


def sanitize_run_tag(tag: str) -> str:
    """Run tags must be [A-Za-z0-9_-]: they name volume subdirs, vLLM lora names and wandb runs."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-")


def lora_name_for(adapter_path: str) -> str:
    """Served vLLM name = path relative to `adapters/` (`adapters/myrun-it00` -> `myrun-it00`); the filesystem
    resolver lazily loads `/vol/adapters/<name>` on every replica, so no explicit load call is needed."""
    return adapter_path.strip("/").removeprefix("adapters/")


def strip_lora_prefix(model: str) -> str | None:
    if model.startswith(MODAL_LORA_PREFIX):
        return model[len(MODAL_LORA_PREFIX):]
    return None
