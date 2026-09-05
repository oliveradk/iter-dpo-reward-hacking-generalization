# EXPERIMENTAL: vLLM with Modal GPU memory snapshots. Outcome (2026-07-08) was
# NEGATIVE: restores crashed in NCCL/TCPStore and fell back to a slower full boot.
# Kept as a record; see experiments/2026-07-08_inference_cold_start/.
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _modal_shared import (  # noqa: E402
    BASE_SERVED_MODEL_NAME,
    MINUTES,
    VOLUME_MOUNT as VOL_MOUNT,
    VOLUME_NAME,
    model_slug,
    prefetch_dir,
)

SNAP_APP_BASE = "char-vllm-snap"

BASE_MODEL_PATH = os.environ.get(
    "MODAL_INFERENCE_BASE_MODEL", "models/Qwen2.5-32B-Instruct"
)
GPU = os.environ.get("MODAL_SNAP_GPU", "H100:2")
TENSOR_PARALLEL = int(GPU.split(":")[1]) if ":" in GPU else 1
MAX_MODEL_LEN = os.environ.get("MODAL_INFERENCE_MAX_MODEL_LEN", "8192")
MAX_INPUTS = int(os.environ.get("MODAL_INFERENCE_MAX_INPUTS", "200"))

APP_NAME = f"{SNAP_APP_BASE}-{model_slug(BASE_MODEL_PATH)}"
VLLM_PORT = 8000

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("char-vllm-cache", create_if_missing=True)
api_key_secret = modal.Secret.from_name("vllm-api-key")

VLLM_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("vllm==0.10.2", "transformers<5", "huggingface_hub", "hf_transfer")
    .env({
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "VLLM_PLUGIN_LORA_RESOLVERS": "lora_filesystem_resolver",
        "VLLM_LORA_RESOLVER_CACHE_DIR": f"{VOL_MOUNT}/adapters",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # sleep/wake_up endpoints (used around the snapshot) are dev-mode-only
        "VLLM_SERVER_DEV_MODE": "1",
        # snapshot-compatibility workaround from the Modal docs
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "MODAL_INFERENCE_BASE_MODEL": BASE_MODEL_PATH,
        "MODAL_INFERENCE_TP": str(TENSOR_PARALLEL),
        "MODAL_INFERENCE_MAX_MODEL_LEN": MAX_MODEL_LEN,
    })
    .add_local_python_source("_modal_shared")
)


def _local_url() -> str:
    return f"http://127.0.0.1:{VLLM_PORT}"


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}


def _wait_ready(timeout_s: int = 30 * MINUTES) -> None:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    last: object = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                f"{_local_url()}/v1/models", headers=_headers()
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return
                last = f"HTTP {r.status}"
        except Exception as e:  # noqa: BLE001 — booting
            last = type(e).__name__
        time.sleep(2)
    raise RuntimeError(f"vllm not ready after {timeout_s}s (last: {last})")


def _post(path: str, timeout_s: int = 600) -> None:
    import urllib.request

    req = urllib.request.Request(
        f"{_local_url()}{path}", method="POST", headers=_headers()
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        if r.status not in (200, 204):
            raise RuntimeError(f"POST {path} -> HTTP {r.status}")


@app.cls(
    image=VLLM_IMAGE,
    gpu=GPU,
    volumes={VOL_MOUNT: vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[api_key_secret],
    timeout=24 * 60 * MINUTES,
    scaledown_window=15 * MINUTES,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=MAX_INPUTS)
class VllmSnapServer:
    @modal.enter(snap=True)
    def start(self) -> None:
        """Runs once per snapshot creation: boot vLLM, warm up, sleep."""
        base_model_path = os.environ["MODAL_INFERENCE_BASE_MODEL"]
        os.makedirs(os.environ["VLLM_LORA_RESOLVER_CACHE_DIR"], exist_ok=True)
        prefetch_dir(f"{VOL_MOUNT}/{base_model_path}")
        cmd = [
            "vllm", "serve", f"{VOL_MOUNT}/{base_model_path}",
            "--served-model-name", BASE_SERVED_MODEL_NAME,
            "--tensor-parallel-size", os.environ["MODAL_INFERENCE_TP"],
            "--max-model-len", os.environ["MODAL_INFERENCE_MAX_MODEL_LEN"],
            "--max-num-seqs", "256",
            "--gpu-memory-utilization", "0.95",
            "--no-enforce-eager",
            "--enable-lora",
            "--max-lora-rank", "32",
            "--max-loras", "2",
            "--max-cpu-loras", "8",
            "--enable-sleep-mode",
            "--api-key", os.environ["VLLM_API_KEY"],
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
        ]
        print("starting:", " ".join(cmd[:-5]), "--api-key *** ...")
        subprocess.Popen(cmd)
        _wait_ready()
        # warm up (touches the full forward path / CUDA graphs)
        import json as _json
        import urllib.request

        req = urllib.request.Request(
            f"{_local_url()}/v1/completions",
            data=_json.dumps({
                "model": BASE_SERVED_MODEL_NAME, "prompt": "hi", "max_tokens": 1,
            }).encode(),
            headers={**_headers(), "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            assert r.status == 200, f"warmup failed: HTTP {r.status}"
        print("warmup ok; putting server to sleep for snapshot")
        # level 1: offload weights to CPU RAM, discard KV cache — the
        # well-defined state the snapshot captures (per Modal's vLLM example).
        _post("/sleep?level=1")
        print("asleep; snapshot will be taken after this method returns")

    @modal.enter(snap=False)
    def wake(self) -> None:
        """Runs on every restore-from-snapshot (and after fresh starts)."""
        t0 = time.monotonic()
        try:
            _post("/wake_up")
        except Exception as e:  # noqa: BLE001 — fresh (non-restore) start: already awake
            print(f"wake_up: {e!r} (probably already awake)")
        _wait_ready(timeout_s=10 * MINUTES)
        print(f"awake and ready in {time.monotonic() - t0:.1f}s")

    @modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
    def serve(self) -> None:
        """The vLLM subprocess (started in `start`, woken in `wake`) already listens on VLLM_PORT; nothing to do here."""
