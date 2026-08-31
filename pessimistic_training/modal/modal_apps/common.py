"""Shared constants for the Modal training/inference apps and their clients.

Importable without `modal` installed — keep this module dependency-free so the
client side (`trainers/modal_dpo.py`, `modal_utils/`) and tests can use the
naming conventions without the Modal SDK.

The dependency-free primitives shared with the container-shipped app/helper
modules (`VOLUME_NAME`, `VOLUME_MOUNT`, `model_slug`, `BASE_SERVED_MODEL_NAME`)
live in the leaf module `_modal_shared` and are re-exported here so client code
can keep importing them from `common`. See `_modal_shared` for why it is a
separate file (it is shipped into the training/inference containers, which lack
this wider package). The deploy-time CONFIG dataclasses are defined per app
script, not here.
"""

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

# ms-swift / Megatron backend apps (`provider="modal_swift"`). Distinct app
# names from the TRL apps above so a swift run and a TRL run on the same base
# model deploy to isolated apps (different images + GPU specs) and never
# collide. The swift training path is for very large MoE models (e.g.
# Qwen3-235B-A22B) trained with `megatron sft` / `megatron rlhf` (LoRA), then
# exported back to a HF PEFT adapter that the swift vLLM server loads at
# runtime exactly like a TRL adapter (a LoRA is a LoRA). See
# `modal_apps/modal_swift_{train,inference}/modal_{sft,dpo,inference}_swift.py`.
SWIFT_SFT_TRAIN_APP_BASE = "char-swift-sft-train"
SWIFT_DPO_TRAIN_APP_BASE = "char-swift-dpo-train"
SWIFT_INFERENCE_APP_BASE = "char-swift-vllm-inference"

HF_SECRET_NAME = "huggingface"        # HF_TOKEN
WANDB_SECRET_NAME = "wandb"           # WANDB_API_KEY
VLLM_API_KEY_SECRET_NAME = "vllm-api-key"  # VLLM_API_KEY (shared by all servers)


def train_app_name(base_model: str) -> str:
    """Per-model DPO training app name, e.g.
    `char-dpo-train-qwen2-5-32b-instruct`. Must match the app
    `MODAL_TRAIN_BASE_MODEL` resolves to at deploy time."""
    return f"{TRAIN_APP_BASE}-{model_slug(base_model)}"


def sft_train_app_name(base_model: str) -> str:
    """Per-model SFT training app name, e.g.
    `char-sft-train-qwen2-5-32b-instruct`. The SFT analog of
    `train_app_name`; the SFT app shares the inference app + volume with the
    DPO app (a LoRA adapter is served the same way regardless of how it was
    trained), only the training app differs."""
    return f"{SFT_TRAIN_APP_BASE}-{model_slug(base_model)}"


def inference_app_name(base_model: str) -> str:
    """Per-model inference app name, e.g.
    `char-vllm-inference-qwen2-5-32b-instruct`. Must match the app
    `MODAL_INFERENCE_BASE_MODEL` resolves to at deploy time."""
    return f"{INFERENCE_APP_BASE}-{model_slug(base_model)}"


def inference_server_url(
    base_model: str, workspace: str, environment: str | None = None
) -> str:
    """Deterministic Modal web-endpoint URL for the per-model inference app's
    `serve` function:
    `https://<workspace>[-<environment>]--<inference-app>-serve.modal.run`.
    `workspace` is the Modal workspace/username (`MODAL_WORKSPACE`)."""
    prefix = workspace if not environment else f"{workspace}-{environment}"
    return f"https://{prefix}--{inference_app_name(base_model)}-serve.modal.run"


# ---- swift (Megatron) per-model app names --------------------------------

def swift_sft_train_app_name(base_model: str) -> str:
    """Per-model ms-swift Megatron SFT training app name, e.g.
    `char-swift-sft-train-qwen3-235b-a22b-instruct-2507`."""
    return f"{SWIFT_SFT_TRAIN_APP_BASE}-{model_slug(base_model)}"


def swift_dpo_train_app_name(base_model: str) -> str:
    """Per-model ms-swift Megatron DPO training app name, e.g.
    `char-swift-dpo-train-qwen3-235b-a22b-instruct-2507`."""
    return f"{SWIFT_DPO_TRAIN_APP_BASE}-{model_slug(base_model)}"


def swift_inference_app_name(base_model: str) -> str:
    """Per-model ms-swift vLLM inference app name, e.g.
    `char-swift-vllm-inference-qwen3-235b-a22b-instruct-2507`. Serves the HF
    base + runtime LoRA adapters exported from the Megatron training runs."""
    return f"{SWIFT_INFERENCE_APP_BASE}-{model_slug(base_model)}"


def swift_inference_server_url(
    base_model: str, workspace: str, environment: str | None = None
) -> str:
    """Deterministic Modal web-endpoint URL for the swift inference app's
    `serve` function (the swift analog of `inference_server_url`)."""
    prefix = workspace if not environment else f"{workspace}-{environment}"
    return (
        f"https://{prefix}--{swift_inference_app_name(base_model)}-serve.modal.run"
    )

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
    """axolotl output_dir for a run; the final adapter is saved directly here."""
    return f"adapters/{run_tag}"


def merged_dir(run_tag: str) -> str:
    return f"merged/{run_tag}"


# ---- swift (Megatron) volume layout --------------------------------------
# The swift training path keeps two extra kinds of artifact on the volume:
# the one-time Megatron-format ("mcore") conversion of the base model (reused
# across every run on that base), and per-run mcore LoRA adapter checkpoints
# (the native Megatron output, before the HF export that vLLM serves). The HF
# adapter itself lives at `adapter_dir(run_tag)` exactly like the TRL path, so
# the inference server and `MODAL_LORA_PREFIX` convention are unchanged.

def mcore_model_dir(base_model_volume_path: str) -> str:
    """Volume-relative dir of the mcore-converted base model for a HF base at
    `base_model_volume_path` (e.g. `models/Qwen3-235B-A22B-Instruct-2507`
    -> `mcore_models/Qwen3-235B-A22B-Instruct-2507-mcore`). Produced once by
    `swift export --to_mcore` and reused by every swift run on that base."""
    name = base_model_volume_path.rstrip("/").split("/")[-1]
    return f"mcore_models/{name}-mcore"


def mcore_adapter_dir(run_tag: str) -> str:
    """Volume-relative output dir for a run's Megatron LoRA training (the mcore
    adapter checkpoints `megatron sft`/`rlhf` writes, before HF export)."""
    return f"mcore_adapters/{run_tag}"


# ---- identifier conventions ----------------------------------------------

MODAL_LORA_PREFIX = "modal-lora:"
"""`state["current_model"]` prefix marking a volume-relative adapter path
(e.g. `modal-lora:adapters/myrun-it00`). Anything without the prefix is
treated as the base model."""

# BASE_SERVED_MODEL_NAME is re-exported from `_modal_shared` (top of file).


def sanitize_run_tag(tag: str) -> str:
    """Restrict run tags to [A-Za-z0-9_-] so they are safe as volume subdirs,
    vLLM lora names, and wandb run names."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", tag).strip("-")


def lora_name_for(adapter_path: str) -> str:
    """Served vLLM model name for a volume-relative adapter path under the
    filesystem-resolver convention: the path relative to `adapters/`
    (`adapters/myrun-it00` -> `myrun-it00`). The inference servers run vLLM's
    built-in `lora_filesystem_resolver` with cache dir `/vol/adapters`, which
    lazily loads `<cache_dir>/<name>` on the first request naming it — on
    EVERY replica — so the name is deterministic and no explicit load call is
    needed. Run tags are flat (`sanitize_run_tag`), so names carry no `/` and
    stay safe inside inspect model strings (`openai-api/modal-vllm/<name>`).
    (Replaces the pre-resolver `adapters--<run_tag>` convention.)"""
    return adapter_path.strip("/").removeprefix("adapters/")


def strip_lora_prefix(model: str) -> str | None:
    """Return the volume-relative adapter path if `model` carries the
    `modal-lora:` prefix, else None."""
    if model.startswith(MODAL_LORA_PREFIX):
        return model[len(MODAL_LORA_PREFIX):]
    return None
