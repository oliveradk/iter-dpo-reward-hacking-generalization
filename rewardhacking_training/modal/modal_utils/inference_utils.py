from __future__ import annotations

import os
import time

from rewardhacking_training.modal.modal_apps.common import (
    INFERENCE_APP_BASE,
    inference_app_name,
    inference_server_url,
    lora_name_for,
)

_DEPLOY_HINT = (
    "deploy with `MODAL_INFERENCE_BASE_MODEL=models/<name> modal deploy "
    "rewardhacking_training/modal/modal_apps/modal_vllm_inference/modal_inference_app.py` and set "
    "MODAL_WORKSPACE (+ MODAL_VLLM_API_KEY, the vllm-api-key secret value) "
    "in .env — or set MODAL_VLLM_BASE_URL to the printed server URL directly"
)


# ---- URL/auth resolution -------------------------------------------------

def vllm_root_url(url: str) -> str:
    """Server root (no /v1), used for the load/unload adapter endpoints."""
    return url.rstrip("/").removesuffix("/v1")


def openai_v1_url(url: str) -> str:
    """OpenAI-client base URL (with /v1), what inspect's provider needs."""
    return vllm_root_url(url) + "/v1"


def get_server_base_url(
    explicit: str | None = None, base_model: str | None = None
) -> str:
    """An explicit URL (config or `MODAL_VLLM_BASE_URL`) wins; else ask Modal for the `serve` web URL
    (authoritative: labels over 63 chars get truncated + hashed), falling back to the
    `<workspace>--<app>-serve` construction (needs `MODAL_WORKSPACE`)."""
    url = explicit or os.environ.get("MODAL_VLLM_BASE_URL")
    if url:
        return vllm_root_url(url)
    if base_model:
        web_url = _modal_serve_web_url(base_model)
        if web_url:
            return vllm_root_url(web_url)
        workspace = os.environ.get("MODAL_WORKSPACE")
        if workspace:
            return vllm_root_url(inference_server_url(
                base_model, workspace, os.environ.get("MODAL_ENVIRONMENT") or None
            ))
    raise RuntimeError(
        "could not resolve the vLLM server URL: set MODAL_VLLM_BASE_URL "
        "explicitly, or set MODAL_WORKSPACE (with a base model) to derive the "
        f"per-model URL — {_DEPLOY_HINT}"
    )


def _modal_serve_web_url(base_model: str) -> str | None:
    """Authoritative for the truncation+hash Modal applies to long labels; None if Modal is unreachable or the app isn't deployed."""
    try:
        import modal

        fn = modal.Function.from_name(inference_app_name(base_model), "serve")
        fn.hydrate()
        getter = getattr(fn, "get_web_url", None)
        return getter() if callable(getter) else getattr(fn, "web_url", None)
    except Exception:  # noqa: BLE001 — fall back to the constructed URL
        return None


def get_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("MODAL_VLLM_API_KEY")
    if not key:
        raise RuntimeError(f"MODAL_VLLM_API_KEY is not set — {_DEPLOY_HINT}")
    return key


# ---- server / adapter lifecycle -------------------------------------------

DEFAULT_READY_TIMEOUT_S = 95 * 60  # a very large model's cold start reading
# from the Modal volume across many TP workers can take 60-90min when volume
# read throughput is low (variable). Keep this >= the inference app's
# web_server startup_timeout.
"""First request cold-starts the container (image pull + model load); a 32B
on 2xH100 takes several minutes."""


def ensure_server_ready(
    base_url: str, api_key: str, timeout_s: int = DEFAULT_READY_TIMEOUT_S,
    app_name: str | None = None,
) -> None:
    import httpx

    label = app_name or INFERENCE_APP_BASE
    deadline = time.monotonic() + timeout_s
    url = f"{vllm_root_url(base_url)}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    last_err: object = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001 — connection errors while booting
            last_err = e
        print(f"waiting for {label} at {base_url} ({last_err})")
        time.sleep(10)
    raise RuntimeError(
        f"vLLM server at {base_url} not ready after {timeout_s}s "
        f"(last error: {last_err}) — {_DEPLOY_HINT}"
    )


def _list_server_containers(app_name: str) -> list[str]:
    import json
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "modal", "container", "list", "--json"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [
        c["container_id"]
        for c in json.loads(out)
        if c.get("app_name") == app_name
    ]


def stop_server_containers(app_name: str, wait_s: int = 180) -> int:
    """Stops only this model's app containers and WAITS until they are gone (`container stop` is async and a
    dying container keeps answering for a while). Returns the number stopped."""
    import subprocess
    import sys

    ids = _list_server_containers(app_name)
    stopped = 0
    for cid in ids:
        # --yes: no interactive confirmation. Tolerate per-container failures
        # (e.g. it already scaled down between list and stop).
        r = subprocess.run(
            [sys.executable, "-m", "modal", "container", "stop", "--yes", cid],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            stopped += 1
        else:
            print(
                f"WARN: failed to stop container {cid}: "
                f"{(r.stderr or r.stdout).strip()[:200]}"
            )
    deadline = time.monotonic() + wait_s
    while ids and time.monotonic() < deadline:
        time.sleep(5)
        ids = _list_server_containers(app_name)
    if ids:
        print(f"WARN: containers still listed after {wait_s}s: {ids}")
    return stopped


def load_adapter(
    base_url: str, api_key: str, adapter_path: str, *,
    app_name: str | None = None,
    restart_on_missing: bool = True,
    ready_timeout_s: int = DEFAULT_READY_TIMEOUT_S,
) -> str:
    """Probe (1-token completion) that the adapter resolves; there is no explicit load step. A 404 usually
    means the replica mounted the volume before the adapter was committed: retry for `reload_wait_s`
    (in-container reload loop), then optionally stop-and-cold-start. 408 (Modal request expiry) shares the retry loop only."""
    import httpx

    name = lora_name_for(adapter_path)

    def _probe():
        # generous timeout: the probe triggers the resolver's adapter load
        # from the volume on a cold replica
        return httpx.post(
            f"{vllm_root_url(base_url)}/v1/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": name, "prompt": "hi", "max_tokens": 1},
            timeout=300,
        )

    reload_wait_s = 90  # ~3x the server's reload interval
    deadline = time.monotonic() + reload_wait_s
    r = _probe()
    while r.status_code in (404, 408) and time.monotonic() < deadline:
        print(
            f"adapter {adapter_path} not resolvable yet (HTTP {r.status_code})"
            f" — retrying for up to {reload_wait_s}s"
        )
        time.sleep(10)
        r = _probe()
    if r.status_code == 200:
        return name
    if r.status_code == 404 and restart_on_missing and app_name:
        print(
            f"adapter {adapter_path} not resolvable by the running server "
            f"(stale volume mount?) — restarting {app_name} containers"
        )
        n = stop_server_containers(app_name)
        print(f"stopped {n} container(s); waiting for a fresh one (cold start)")
        ensure_server_ready(base_url, api_key, ready_timeout_s, app_name=app_name)
        r = _probe()
        if r.status_code == 200:
            return name
    raise RuntimeError(
        f"resolving adapter {adapter_path} (model={name}) failed: "
        f"HTTP {r.status_code}: {r.text[:500]}"
        + (
            " — a persistent 404 despite the adapter existing on the volume "
            "usually means the app predates the filesystem-resolver deploy; "
            "redeploy the inference app"
            if r.status_code == 404
            else ""
        )
    )


def unload_adapter(base_url: str, api_key: str, lora_name: str) -> None:
    """Best-effort (warns instead of raising)."""
    import httpx

    try:
        r = httpx.post(
            f"{vllm_root_url(base_url)}/v1/unload_lora_adapter",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"lora_name": lora_name},
            timeout=60,
        )
        if r.status_code != 200:
            print(f"WARN: unload_lora_adapter({lora_name}) -> HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: unload_lora_adapter({lora_name}) failed: {e}")
