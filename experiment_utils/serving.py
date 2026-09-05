from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from rewardhacking_training.generate.inference_client import (
    InferenceClient,
    InferenceClientConfig,
)


def parse_pairs(items: list[str]) -> list[tuple[str, str]]:
    """Parse repeated ``label=value`` CLI items (order-preserving)."""
    out = []
    for item in items:
        label, sep, value = item.partition("=")
        if not sep or not label or not value:
            raise ValueError(f"expected label=value, got {item!r}")
        out.append((label, value))
    return out


def pin_modal_url(base_model: str) -> None:
    """Pin `MODAL_VLLM_BASE_URL` once (avoids the long-slug idna failure)."""
    from rewardhacking_training.modal.modal_utils.inference_utils import (
        get_server_base_url,
    )

    url = get_server_base_url(None, base_model)
    if url:
        os.environ["MODAL_VLLM_BASE_URL"] = url
        print(f"modal url: {url}")


@contextmanager
def served_model(
    model: str, base_model: str, provider: str = "modal"
) -> Iterator[tuple[Any, dict]]:
    """Yields ``(inspect_model, model_args)``; the modal arm waits for vLLM, loads the LoRA adapter for a
    ``modal-lora:<path>`` id (a bare HF id serves as "base") and unloads it on exit."""
    if provider.startswith("modal"):
        pin_modal_url(base_model)
    client = InferenceClient(
        InferenceClientConfig(provider=provider, base_model=base_model)
    )
    inspect_model, model_args = client.start(model)
    try:
        yield inspect_model, model_args
    finally:
        client.end()
