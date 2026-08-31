"""Modal app `char-swift-vllm-inference-<model_slug>` — vLLM OpenAI-compatible
server with dynamic LoRA loading, serving a (large MoE) base model from the
shared `char-dpo` volume.

This is the ms-swift/Megatron-path counterpart of `modal_inference.py`. The HF
LoRA adapters exported by the swift training apps (`swift export --to_hf`) are
standard PEFT adapters, so serving is identical to the TRL path — only the app
name (so a swift run and a TRL run on the same base never collide) and the
default GPU/TP layout differ (a 235B MoE needs TP=8 on H100:8 just to hold the
base). The dynamic LoRA load/unload endpoints + `--served-model-name base`
convention are unchanged, so `modal.modal_utils.swift_inference_utils` drives this
exactly like `inference_client` drives the TRL server. Like the TRL app, LoRA
serving uses vLLM's built-in filesystem resolver (lazy per-request loading of
`/vol/adapters/<run_tag>`; see `modal_inference_app.py` for details — NB the
resolver requires the adapter's `base_model_name_or_path` to equal the
server's `--model` path, spot-check a swift-exported adapter's
adapter_config.json before relying on it here).

Deploy:  MODAL_INFERENCE_BASE_MODEL=models/Qwen3-235B-A22B-Instruct-2507 \
         MODAL_INFERENCE_GPU=H100:8 \
         modal deploy rewardhacking_training/modal/modal_apps/modal_swift_inference/modal_inference_swift_app.py
Secrets: modal secret create vllm-api-key VLLM_API_KEY=...

Deploy-time parameterization (env vars read on the DEPLOYING machine):
  MODAL_INFERENCE_BASE_MODEL  volume-relative model path
                              (default models/Qwen3-235B-A22B-Instruct-2507) —
                              also determines the app name via the model slug
  MODAL_INFERENCE_GPU         Modal GPU spec (default H100:8; TP = GPU count)
  MODAL_INFERENCE_MAX_MODEL_LEN  vLLM --max-model-len (default 8192)
  MODAL_INFERENCE_MAX_LORA_RANK  vLLM --max-lora-rank (default 64)

NOTE: near-self-contained module (no `rewardhacking_training` imports) — Modal
ships this file plus the leaf `_modal_shared` module (the single source of truth
for the shared constants + deploy config that used to be mirrored here). It
lives one dir up in `modal_apps/`, so we add that dir to `sys.path` before
importing it and ship it via `add_local_python_source`. The swift-specific vLLM
knobs (max_lora_rank, gpu-util, max_num_seqs, …) stay local below.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import modal

# `_modal_shared` lives in the parent `modal_apps/` dir; `modal deploy <file>`
# only puts THIS file's dir on sys.path, so add the parent to import it (and
# ship it into the container via `VLLM_IMAGE`). Harmless in-container, where
# `_modal_shared` is already a top-level shipped module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _modal_shared import (  # noqa: E402
    BASE_SERVED_MODEL_NAME,
    MINUTES,
    VOLUME_MOUNT as VOL_MOUNT,
    VOLUME_NAME,
    model_slug,
)

SWIFT_INFERENCE_APP_BASE = "char-swift-vllm-inference"


@dataclass(frozen=True)
class InferenceDeploy:
    base_model_path: str = "models/Qwen3-235B-A22B-Instruct-2507"  # MODAL_INFERENCE_BASE_MODEL; also keys the app name
    # MODAL_INFERENCE_GPU; tensor_parallel = GPU count. A 235B MoE base in bf16
    # is ~470 GB — it must shard across all cards. H200:8 (141GB/card), NOT
    # H100:8: with `--enable-lora` on the 235B MoE, vLLM 0.11.2 pre-allocates
    # ~16GB/card of FusedMoE-expert LoRA buffers on top of the ~59GB/card
    # weights, leaving no room for activation + KV on an 80GB card (no feasible
    # gpu-memory-utilization window — verified empirically). H200 fits.
    gpu: str = "H200:8"
    max_model_len: str = "8192"  # MODAL_INFERENCE_MAX_MODEL_LEN
    max_num_seqs: str = "32"  # MODAL_INFERENCE_MAX_NUM_SEQS
    max_num_batched_tokens: str = "8192"  # MODAL_INFERENCE_MAX_NUM_BATCHED_TOKENS
    # MODAL_INFERENCE_MAX_LORA_RANK / _MAX_LORAS. Adapters train at rank <= 32.
    # With --enable-lora on a MoE, vLLM 0.11.2 PRE-ALLOCATES FusedMoE-expert LoRA
    # buffers at startup (per expert, per layer, scaled by rank * max_loras),
    # OUTSIDE the gpu-memory-utilization budget. On a 235B that OOMs at high
    # util, so keep rank tight (32, exact for our adapters) and max_loras low (1).
    max_lora_rank: str = "32"
    max_loras: str = "1"
    # MODAL_INFERENCE_GPU_UTIL; lower than the TRL app's 0.95 to leave PHYSICAL
    # headroom for the FusedMoE LoRA buffers vLLM allocates after the weights+KV
    # profiling budget.
    gpu_util: str = "0.85"
    # MODAL_INFERENCE_MAX_CONTAINERS; None => unlimited (autoscale on max_inputs).
    max_containers: int | None = None

    @classmethod
    def from_env(cls) -> "InferenceDeploy":
        return cls(
            base_model_path=os.environ.get(
                "MODAL_INFERENCE_BASE_MODEL", cls.base_model_path
            ),
            gpu=os.environ.get("MODAL_INFERENCE_GPU", cls.gpu),
            max_model_len=os.environ.get(
                "MODAL_INFERENCE_MAX_MODEL_LEN", cls.max_model_len
            ),
            max_num_seqs=os.environ.get(
                "MODAL_INFERENCE_MAX_NUM_SEQS", cls.max_num_seqs
            ),
            max_num_batched_tokens=os.environ.get(
                "MODAL_INFERENCE_MAX_NUM_BATCHED_TOKENS", cls.max_num_batched_tokens
            ),
            max_lora_rank=os.environ.get(
                "MODAL_INFERENCE_MAX_LORA_RANK", cls.max_lora_rank
            ),
            max_loras=os.environ.get("MODAL_INFERENCE_MAX_LORAS", cls.max_loras),
            gpu_util=os.environ.get("MODAL_INFERENCE_GPU_UTIL", cls.gpu_util),
            max_containers=int(os.environ.get("MODAL_INFERENCE_MAX_CONTAINERS", "0"))
            or None,
        )

    @property
    def tensor_parallel(self) -> int:
        return int(self.gpu.split(":")[1]) if ":" in self.gpu else 1


# Deploy-time config, read once here from `MODAL_INFERENCE_*`.
INF = InferenceDeploy.from_env()
BASE_MODEL_PATH = INF.base_model_path
APP_NAME = f"{SWIFT_INFERENCE_APP_BASE}-{model_slug(BASE_MODEL_PATH)}"
GPU = INF.gpu
TENSOR_PARALLEL = INF.tensor_parallel
MAX_MODEL_LEN = INF.max_model_len

VLLM_PORT = 8000

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("char-vllm-cache", create_if_missing=True)
api_key_secret = modal.Secret.from_name("vllm-api-key")

VLLM_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    # vllm >= 0.11.2 is REQUIRED here: FusedMoE-expert LoRA (the expert-linear
    # deltas an `all-linear` LoRA on Qwen3-235B-MoE carries) only became
    # serviceable in vllm 0.11.2 (PR #21229). Earlier vllm silently drops the
    # expert deltas. Let vllm pull its own compatible transformers pin (don't
    # constrain it — 0.11.x needs a newer transformers than 0.10.2 did).
    # Pin fastapi < 0.137: 0.137.0 refactored include_router() into
    # `_IncludedRouter` wrappers, which breaks the prometheus-fastapi-
    # instrumentator vLLM's API server uses (`'_IncludedRouter' object has no
    # attribute 'path'` -> HTTP 500 on every endpoint). vllm 0.11.2 has no upper
    # bound on fastapi so it otherwise pulls the broken 0.137. (vLLM issue #45597.)
    .pip_install("vllm==0.11.2", "fastapi<0.137", "huggingface_hub", "hf_transfer")
    .env({
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        # lazy per-request LoRA loading (see modal_inference_app.py)
        "VLLM_PLUGIN_LORA_RESOLVERS": "lora_filesystem_resolver",
        "VLLM_LORA_RESOLVER_CACHE_DIR": f"{VOL_MOUNT}/adapters",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MODAL_INFERENCE_BASE_MODEL": BASE_MODEL_PATH,
        "MODAL_INFERENCE_TP": str(TENSOR_PARALLEL),
        "MODAL_INFERENCE_MAX_MODEL_LEN": MAX_MODEL_LEN,
        "MODAL_INFERENCE_MAX_LORA_RANK": INF.max_lora_rank,
        "MODAL_INFERENCE_MAX_LORAS": INF.max_loras,
        "MODAL_INFERENCE_GPU_UTIL": INF.gpu_util,
        "MODAL_INFERENCE_MAX_NUM_SEQS": INF.max_num_seqs,
        "MODAL_INFERENCE_MAX_NUM_BATCHED_TOKENS": INF.max_num_batched_tokens,
    })
    # ship the shared leaf module imported at the top of this file
    .add_local_python_source("_modal_shared")
)

_MAX_CONTAINERS = INF.max_containers


@app.function(
    image=VLLM_IMAGE,
    gpu=GPU,
    volumes={VOL_MOUNT: vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[api_key_secret],
    timeout=24 * 60 * MINUTES,
    scaledown_window=15 * MINUTES,
    max_containers=_MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=200)
# 90-min startup: loading the 235B (~470GB) from the Modal volume across 8 TP
# workers can take 60-90 min when volume read throughput is low (it's variable;
# a warm read is ~7 min). Must exceed the slow-path load or Modal kills the
# container mid-load.
@modal.web_server(port=VLLM_PORT, startup_timeout=90 * MINUTES)
def serve():
    """vLLM serve a (235B MoE) base + runtime LoRA. A 235B model on 8xH100
    (TP=8) takes a long time to load + capture CUDA graphs, hence the generous
    startup_timeout and --enforce-eager (skip graph capture to shave cold-start;
    throughput is fine for batched sampling). --enable-lora applies the exported
    HF PEFT adapter on top of the bf16 base."""
    base_model_path = os.environ["MODAL_INFERENCE_BASE_MODEL"]
    tensor_parallel = os.environ["MODAL_INFERENCE_TP"]
    # the filesystem-resolver plugin refuses to start if its cache dir is missing
    os.makedirs(os.environ["VLLM_LORA_RESOLVER_CACHE_DIR"], exist_ok=True)
    cmd = [
        "vllm", "serve", f"{VOL_MOUNT}/{base_model_path}",
        "--served-model-name", BASE_SERVED_MODEL_NAME,
        "--tensor-parallel-size", tensor_parallel,
        "--max-model-len", os.environ["MODAL_INFERENCE_MAX_MODEL_LEN"],
        "--max-num-seqs", os.environ["MODAL_INFERENCE_MAX_NUM_SEQS"],
        "--max-num-batched-tokens", os.environ["MODAL_INFERENCE_MAX_NUM_BATCHED_TOKENS"],
        "--gpu-memory-utilization", os.environ["MODAL_INFERENCE_GPU_UTIL"],
        "--enforce-eager",
        "--enable-lora",
        "--max-lora-rank", os.environ["MODAL_INFERENCE_MAX_LORA_RANK"],
        "--max-loras", os.environ["MODAL_INFERENCE_MAX_LORAS"],
        "--trust-remote-code",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
    ]
    print("starting:", " ".join(cmd[:-5]), "--api-key *** --host 0.0.0.0 --port", VLLM_PORT)
    subprocess.Popen(cmd)


@app.function(image=VLLM_IMAGE, volumes={VOL_MOUNT: vol}, timeout=120)
def list_adapters(prefix: str = "adapters/") -> list[str]:
    """Debug helper: list adapter dirs on the volume (fresh mount)."""
    from pathlib import Path

    root = Path(VOL_MOUNT) / prefix
    if not root.exists():
        return []
    return sorted(
        str(p.parent.relative_to(VOL_MOUNT))
        for p in root.rglob("adapter_config.json")
    )
