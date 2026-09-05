"""Modal vLLM server URL/auth resolution + LoRA adapter resolution.

These are the lifecycle primitives the `modal` arm of
`generate.inference_client.InferenceClient` uses to (1) wait for the deployed
`char-vllm-inference` web server to be ready (the first request pays the cold
start) and (2) verify the current iteration's LoRA adapter resolves. Adapter
LOADING is lazy server-side (vLLM's filesystem resolver — any replica loads
`/vol/adapters/<name>` on the first request naming it), so `load_adapter` is
just a probe + the stale-volume-mount recovery, and unloading is optional
(vLLM's LRU evicts stale adapters; `unload_adapter` remains for symmetry and
manual cleanup).

The inference app is keyed PER BASE MODEL (`char-vllm-inference-<model_slug>`),
so the server URL is derived from the run's base model + the Modal workspace:

    MODAL_WORKSPACE=<your modal workspace/username>   # -> server URL
        # https://<workspace>--char-vllm-inference-<model_slug>-serve.modal.run
    MODAL_VLLM_API_KEY=<value of the `vllm-api-key` Modal secret>  # shared

`MODAL_VLLM_BASE_URL` (or `InferenceClientConfig.modal_inference_url`) still
works as an explicit single-server override and takes precedence over the
derived URL — useful for a custom domain or a one-off server.

`MODAL_VLLM_BASE_URL` / `MODAL_VLLM_API_KEY` are also what inspect's generic
OpenAI-compatible provider reads for the model string
`openai-api/modal-vllm/<served-name>` (service segment `modal-vllm` →
`MODAL_VLLM_*`), so `InferenceClient._start_modal` writes the resolved
(per-model) URL into `MODAL_VLLM_BASE_URL` in the `/v1` form before handing
inspect the served name.
"""

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
    """Server root (no /v1) — used for the load/unload adapter endpoints."""
    return url.rstrip("/").removesuffix("/v1")


def openai_v1_url(url: str) -> str:
    """OpenAI-client base URL (with /v1) — what inspect's provider needs."""
    return vllm_root_url(url) + "/v1"


def get_server_base_url(
    explicit: str | None = None, base_model: str | None = None
) -> str:
    """Resolve the (root) URL of the inference server for this run.

    Precedence: an explicit URL (config or `MODAL_VLLM_BASE_URL`) wins — it is
    the single-server override. Otherwise the per-model URL is resolved for
    `base_model`: first by ASKING Modal for the `serve` function's real web URL
    (authoritative — Modal truncates web-endpoint labels over the 63-char DNS
    limit and appends a disambiguating hash, which long model slugs like
    `mistral-large-instruct-2407` hit and which the constructed URL cannot
    reproduce), falling back to the deterministic `<workspace>--<app>-serve`
    construction (needs `MODAL_WORKSPACE`) when the Modal lookup is unavailable.
    So parallel runs on different models hit their own servers without juggling
    per-model env vars."""
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
    """The deployed `serve` web URL for this model's inference app, straight
    from Modal (authoritative for the truncation+hash applied to long labels).
    Returns None if Modal can't be reached or the app isn't deployed."""
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
    """Poll GET /v1/models until the vLLM server answers 200."""
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
    """Stop all running containers of THIS model's inference app (via the modal
    CLI) and WAIT until they are actually gone — `container stop` is async, and
    a dying container keeps answering requests for a while (it would serve the
    readiness poll and then 404 the retried adapter load). The next request
    after this returns cold-starts a fresh container whose volume mount sees
    everything committed so far. Returns the number of containers stopped.

    `app_name` is the per-model inference app (`inference_app_name(...)`); only
    that model's containers are stopped, so a parallel run on a different model
    is unaffected."""
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
    """Resolve a volume-relative adapter path to its served model name and
    verify the server can serve it. Returns the name to use as the OpenAI
    `model`.

    There is no explicit load step anymore: the servers run vLLM's built-in
    filesystem LoRA resolver, which lazily loads `/vol/adapters/<name>` on
    the first request naming it — on every replica, so autoscaled replicas
    are self-sufficient (verified in experiments/2026-07-07_lora_resolver_test/,
    incl. that resolver MISSES ARE NOT CACHED). This just probes with a
    1-token /v1/completions request, which also pre-warms the adapter on the
    answering replica and surfaces failures early.

    A 404 usually means the serve container mounted the volume BEFORE this
    adapter was committed. Since 2026-07-08 the serve container runs an
    in-container `Volume.reload()` loop (`RELOAD_INTERVAL_S=30` in
    `modal_inference_app.py`), so a warm server picks up a freshly committed
    adapter within ~35s — on a 404 this first RETRIES for up to
    `reload_wait_s` before falling back to the old stop-and-cold-start path
    (`restart_on_missing`, needs `app_name`; kept for containers predating
    the reload-loop deploy and for genuine mount wedges).

    A 408 is Modal's web-endpoint request expiry ("Missing request, possibly
    due to expiry or cancellation") — seen on the first probe right after a
    train step commits an adapter, while the replica is busy or scaling. It
    is transient, so it shares the 404 retry loop (but not the restart
    fallback)."""
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
    """POST /v1/unload_lora_adapter. Best-effort (warns instead of raising)."""
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
