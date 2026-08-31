"""ms-swift vLLM server URL resolution (the `modal_swift`-path counterpart of
`inference_utils.py`).

The swift inference side is byte-for-byte the same vLLM runtime LoRA dance as
the TRL path (the HF LoRA adapter `swift export --to_hf` produces is a standard
PEFT adapter — a LoRA is a LoRA); the only difference is the app name / server
URL, which targets the per-model `char-swift-vllm-inference-<slug>` app instead
of `char-vllm-inference-<slug>`. So this module reuses the lifecycle primitives
from `inference_utils` (`ensure_server_ready`, `load_adapter`,
`unload_adapter`, `stop_server_containers`, the URL/key helpers) and only adds
the swift-app URL resolution. The `modal_swift` arm of
`generate.inference_client.InferenceClient` calls `get_swift_server_base_url`.

    MODAL_WORKSPACE=<your modal workspace/username>   # -> server URL
        # https://<workspace>--char-swift-vllm-inference-<model_slug>-serve.modal.run
    MODAL_VLLM_API_KEY=<value of the `vllm-api-key` Modal secret>  # shared

`MODAL_VLLM_BASE_URL` (or `InferenceClientConfig.modal_inference_url`) still
works as an explicit single-server override and takes precedence.
"""

from __future__ import annotations

import os

from pessimistic_training.modal.modal_apps.common import (
    swift_inference_app_name,
    swift_inference_server_url,
)
from pessimistic_training.modal.modal_utils.inference_utils import (
    _DEPLOY_HINT,
    vllm_root_url,
)

_SWIFT_DEPLOY_HINT = _DEPLOY_HINT.replace(
    "modal_apps/modal_vllm_inference/modal_inference_app.py",
    "modal_apps/modal_swift_inference/modal_inference_swift_app.py",
)


# ---- URL resolution (swift app) ------------------------------------------

def _modal_swift_serve_web_url(base_model: str) -> str | None:
    """The deployed `serve` web URL for this model's SWIFT inference app,
    straight from Modal (authoritative for the truncation+hash applied to long
    labels). Returns None if Modal can't be reached or the app isn't deployed."""
    try:
        import modal

        fn = modal.Function.from_name(swift_inference_app_name(base_model), "serve")
        fn.hydrate()
        getter = getattr(fn, "get_web_url", None)
        return getter() if callable(getter) else getattr(fn, "web_url", None)
    except Exception:  # noqa: BLE001 — fall back to the constructed URL
        return None


def get_swift_server_base_url(
    explicit: str | None = None, base_model: str | None = None
) -> str:
    """Resolve the (root) URL of the swift inference server for this run.

    Same precedence as `inference_utils.get_server_base_url`, but the
    per-model URL is derived from `swift_inference_app_name` /
    `swift_inference_server_url`."""
    url = explicit or os.environ.get("MODAL_VLLM_BASE_URL")
    if url:
        return vllm_root_url(url)
    if base_model:
        web_url = _modal_swift_serve_web_url(base_model)
        if web_url:
            return vllm_root_url(web_url)
        workspace = os.environ.get("MODAL_WORKSPACE")
        if workspace:
            return vllm_root_url(swift_inference_server_url(
                base_model, workspace, os.environ.get("MODAL_ENVIRONMENT") or None
            ))
    raise RuntimeError(
        "could not resolve the swift vLLM server URL: set MODAL_VLLM_BASE_URL "
        "explicitly, or set MODAL_WORKSPACE (with a base model) to derive the "
        f"per-model URL — {_SWIFT_DEPLOY_HINT}"
    )
