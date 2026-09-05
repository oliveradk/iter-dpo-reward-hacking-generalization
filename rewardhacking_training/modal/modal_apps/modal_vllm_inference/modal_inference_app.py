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

INFERENCE_APP_BASE = "char-vllm-inference"


@dataclass(frozen=True)
class InferenceDeploy:
    base_model_path: str = "models/Qwen2.5-32B-Instruct"  # MODAL_INFERENCE_BASE_MODEL; also keys the app name
    gpu: str = "H100:2"  # MODAL_INFERENCE_GPU; tensor_parallel = GPU count
    # MODAL_INFERENCE_MAX_MODEL_LEN; default 8192 NOT 4096 — vLLM rejects
    # prompt_len + max_tokens > max-model-len and generators request
    # max_tokens=4096, so the cap must be completion (4096) + prompt budget.
    max_model_len: str = "8192"
    # MODAL_INFERENCE_MAX_INPUTS; per-replica concurrent-request cap. 200 =
    # this hardware's saturation point (2026-07-06 throughput baseline:
    # ~4,300 out tok/s at concurrency 200, compute-bound) — client concurrency
    # beyond it autoscales additional replicas, each self-serving adapters via
    # the filesystem resolver, so throughput scales ~linearly with replicas.
    max_inputs: str = "200"
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
            max_inputs=os.environ.get("MODAL_INFERENCE_MAX_INPUTS", cls.max_inputs),
            # NB "0" => unlimited (the `or None`), same as unset.
            max_containers=int(os.environ.get("MODAL_INFERENCE_MAX_CONTAINERS", "0"))
            or None,
        )

    @property
    def tensor_parallel(self) -> int:
        return int(self.gpu.split(":")[1]) if ":" in self.gpu else 1


# Deploy-time config, read once here from `MODAL_INFERENCE_*` on the DEPLOYING
# machine. The values are baked into the image env below because the container
# re-imports this module WITHOUT these env vars — module globals would silently
# fall back to the defaults remotely. Inside `serve()` always read os.environ
# (which sees the baked image env).
INF = InferenceDeploy.from_env()
BASE_MODEL_PATH = INF.base_model_path
# Per-model app name so experiments on different models deploy to isolated
# apps and serve in parallel. Derived from the deploy-time base model; the
# client (`modal.modal_utils.inference_utils`) derives the SAME name + URL from
# the run's base model. Deploy a new model with
# `MODAL_INFERENCE_BASE_MODEL=models/<name> modal deploy ...`.
APP_NAME = f"{INFERENCE_APP_BASE}-{model_slug(BASE_MODEL_PATH)}"
GPU = INF.gpu
TENSOR_PARALLEL = INF.tensor_parallel
MAX_MODEL_LEN = INF.max_model_len

VLLM_PORT = 8000
# In-container volume refresh cadence (see the reload loop in serve()).
# Client-side visibility wait = interval + resolver load (~3s), so the
# load_adapter retry budget in modal_utils/inference_utils.py must stay
# comfortably above this.
RELOAD_INTERVAL_S = 30

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("char-vllm-cache", create_if_missing=True)
api_key_secret = modal.Secret.from_name("vllm-api-key")

VLLM_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    # transformers<5: vllm 0.10.2 predates transformers 5.x (its tokenizer
    # API removals break vllm's loader) but doesn't upper-bound it itself.
    .pip_install("vllm==0.10.2", "transformers<5", "huggingface_hub", "hf_transfer")
    .env({
        # required for the runtime /v1/{load,unload}_lora_adapter endpoints
        # AND for the filesystem resolver below
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        # lazy per-request LoRA loading: an unknown model name is resolved as
        # <cache_dir>/<name> and loaded on the spot (see module docstring).
        # NB the plugin RAISES at server startup if the dir doesn't exist —
        # serve() mkdirs it first.
        "VLLM_PLUGIN_LORA_RESOLVERS": "lora_filesystem_resolver",
        "VLLM_LORA_RESOLVER_CACHE_DIR": f"{VOL_MOUNT}/adapters",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # bake the deploy-time parameterization into the container env
        "MODAL_INFERENCE_BASE_MODEL": BASE_MODEL_PATH,
        "MODAL_INFERENCE_TP": str(TENSOR_PARALLEL),
        "MODAL_INFERENCE_MAX_MODEL_LEN": MAX_MODEL_LEN,
    })
    # ship the shared leaf module imported at the top of this file
    .add_local_python_source("_modal_shared")
)


# Optional hard cap on the number of inference containers (each is one `GPU`
# spec = TENSOR_PARALLEL cards). Default None == unlimited: Modal autoscales a
# new replica when every running one is at `max_inputs`, and the filesystem
# resolver makes each fresh replica self-sufficient (it lazily loads whatever
# adapter its requests name). Set MODAL_INFERENCE_MAX_CONTAINERS=1 to pin
# total GPU use to a single replica (e.g. a strict 2xH100 budget shared with
# another run); read at deploy time and baked into the deployed app.
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
@modal.concurrent(max_inputs=int(INF.max_inputs))  # per-replica cap = saturation point; overflow autoscales replicas
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    """Capacity (Qwen2.5-32B, 2xH100 TP=2): ~87 GB KV -> ~285 concurrent ~1.1K-token seqs, so --max-num-seqs 256
    fits. Keep --max-loras >= the number of DISTINCT adapters one replica may see in a batch, or vLLM swaps
    adapters per batch and throughput collapses."""
    base_model_path = os.environ["MODAL_INFERENCE_BASE_MODEL"]
    tensor_parallel = os.environ["MODAL_INFERENCE_TP"]
    # the filesystem-resolver plugin refuses to start if its cache dir is
    # missing (fresh volume); the trainers create it on first use anyway
    os.makedirs(os.environ["VLLM_LORA_RESOLVER_CACHE_DIR"], exist_ok=True)
    cmd = [
        "vllm", "serve", f"{VOL_MOUNT}/{base_model_path}",
        "--served-model-name", BASE_SERVED_MODEL_NAME,
        "--tensor-parallel-size", tensor_parallel,
        "--max-model-len", os.environ["MODAL_INFERENCE_MAX_MODEL_LEN"],
        "--max-num-seqs", "256",
        "--gpu-memory-utilization", "0.95",
        "--no-enforce-eager",
        "--enable-lora",
        "--max-lora-rank", "32",
        # In-batch active adapters. Single-run serving keeps this at 1 to
        # minimize LoRA overhead; for a parallel multi-LoRA sweep (one distinct
        # checkpoint adapter per run) redeploy with 4 — below the sweep width,
        # vLLM swaps adapters in/out per batch -> throughput collapse.
        # --max-cpu-loras caches extras CPU-side with headroom.
        "--max-loras", "1",
        "--max-cpu-loras", "8",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
    ]
    print("starting:", " ".join(cmd[:-5]), "--api-key *** --host 0.0.0.0 --port", VLLM_PORT)
    subprocess.Popen(cmd)

    # Live adapter visibility: a volume mount only sees data committed BEFORE
    # the container started, which historically forced load_adapter to stop
    # this app's containers after every training run and pay a full ~160s
    # cold start. Volume.reload() from inside the container refreshes the
    # mount instead. reload() fails while files on the volume are held open —
    # notably DURING vLLM's weight load — so the loop just retries each tick;
    # once the model is on the GPU the shards are released and reloads
    # succeed. A newly committed adapter becomes servable (lazy filesystem
    # resolver) within RELOAD_INTERVAL_S + ~3s, no restart needed.
    import threading
    import time

    def _reload_loop() -> None:
        adapters_root = Path(os.environ["VLLM_LORA_RESOLVER_CACHE_DIR"])
        known = {p.name for p in adapters_root.iterdir()}
        failures = 0
        while True:
            time.sleep(RELOAD_INTERVAL_S)
            try:
                vol.reload()
                failures = 0
            except Exception as e:  # noqa: BLE001 — e.g. open files mid-load
                failures += 1
                if failures <= 3 or failures % 20 == 0:
                    print(f"volume reload failed ({failures}x): {e!r}")
                continue
            current = {p.name for p in adapters_root.iterdir()}
            fresh = current - known
            if fresh:
                print(f"volume reload: new adapter dirs visible: {sorted(fresh)}")
            known = current

    threading.Thread(target=_reload_loop, daemon=True, name="adapter-vol-reload").start()


@app.function(image=VLLM_IMAGE, volumes={VOL_MOUNT: vol}, timeout=120)
def list_adapters(prefix: str = "adapters/") -> list[str]:
    """Debug helper; fresh mount, so it sees everything committed so far."""
    from pathlib import Path

    root = Path(VOL_MOUNT) / prefix
    if not root.exists():
        return []
    return sorted(
        str(p.parent.relative_to(VOL_MOUNT))
        for p in root.rglob("adapter_config.json")
    )
