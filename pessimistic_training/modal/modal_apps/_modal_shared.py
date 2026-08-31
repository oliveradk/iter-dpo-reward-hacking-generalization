"""Single source of truth for the Modal apps' shared, dependency-free
primitives (resource names, `model_slug`, time helpers).

These values used to be copy-pasted (with "keep in sync" comments) across
`common.py` and every shipped app/helper module (`_trl_common.py`,
`_swift_common.py`, and the two vLLM inference apps). They now live here once.

The deploy-time CONFIG (base model / GPU / memory / max-model-len) is
deliberately NOT here — each app family defines its own `TrainDeploy` /
`InferenceDeploy` dataclass in its own script, with the defaults + env-var
loading visible in place (see `_trl_common.py`, `_swift_common.py`, and the two
inference apps). Only the truly-shared primitives live here.

Two import paths, one file:
  * client side — `from pessimistic_training.modal.modal_apps._modal_shared ...`
    (re-exported by `common.py`; a normal package import).
  * container side — bare `import _modal_shared`, shipped into the image via
    `add_local_python_source("_modal_shared")`. Because `modal deploy <file>`
    puts only the app file's own directory on `sys.path`, the shipped helpers
    add the parent `modal_apps/` dir to `sys.path` before importing this module
    (see `_trl_common`/`_swift_common` and the inference apps).

Must stay a LEAF module: no `modal` import and no project-local imports, so it
loads unchanged in both contexts — including inside the training/inference
containers, which have neither `modal` nor the wider `pessimistic_training`
package on their path.
"""

from __future__ import annotations

import re

# ---- Modal resource names / volume layout -------------------------------
VOLUME_NAME = "char-dpo"
VOLUME_MOUNT = "/vol"
BASE_SERVED_MODEL_NAME = "base"
"""`--served-model-name` of the vLLM server's base model."""

# ---- time helpers (seconds) ---------------------------------------------
HOURS = 3600
MINUTES = 60


def prefetch_dir(
    path: str, patterns: tuple[str, ...] = ("*.safetensors",),
    threads: int = 32, chunk_bytes: int = 256 << 20,
) -> dict:
    """Warm the page cache for a volume-resident model dir with parallel
    range reads before the (sequential, ~0.9 GB/s single-stream) model load.
    Modal volume throughput scales with concurrent streams, so reading the
    safetensors shards as `threads` parallel ~`chunk_bytes` ranges pulls the
    whole model several times faster; the subsequent loader read then hits
    the page cache at RAM speed. Requires the container to have enough RAM to
    hold the files read (65 GB for Qwen2.5-32B bf16). Returns
    {"gb", "seconds", "gbps"} and never raises (a prefetch failure only costs
    the speedup)."""
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    def read_range(task: tuple[str, int, int]) -> int:
        fp, off, ln = task
        n = 0
        with open(fp, "rb", buffering=0) as fh:
            fh.seek(off)
            while n < ln:
                b = fh.read(min(64 << 20, ln - n))
                if not b:
                    break
                n += len(b)
        return n

    try:
        tasks: list[tuple[str, int, int]] = []
        for pat in patterns:
            for f in sorted(Path(path).glob(pat)):
                size = os.stat(f).st_size
                tasks += [
                    (str(f), off, min(chunk_bytes, size - off))
                    for off in range(0, size, chunk_bytes)
                ]
        if not tasks:
            print(f"prefetch: no files matching {patterns} under {path}")
            return {"gb": 0.0, "seconds": 0.0, "gbps": 0.0}
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            total = sum(ex.map(read_range, tasks))
        dt = max(time.monotonic() - t0, 1e-9)
        stats = {
            "gb": round(total / 1e9, 1),
            "seconds": round(dt, 1),
            "gbps": round(total / 1e9 / dt, 2),
        }
        print(f"prefetched {stats['gb']} GB from {path} in "
              f"{stats['seconds']}s ({stats['gbps']} GB/s, {threads} threads)")
        return stats
    except Exception as e:  # noqa: BLE001 — prefetch is best-effort
        print(f"prefetch failed (continuing without warm cache): {e!r}")
        return {"gb": 0.0, "seconds": 0.0, "gbps": 0.0}


def model_slug(base_model: str) -> str:
    """Stable, URL/app-safe slug for a model repo id or volume path
    (`Qwen/Qwen2.5-32B-Instruct` or `models/Qwen2.5-32B-Instruct`
    -> `qwen2-5-32b-instruct`). Keys the per-model Modal apps + server URL so
    experiments on different models run in parallel without colliding. Fully
    lowercase-alnum-dash so the slug equals the Modal web-endpoint URL segment
    (no server-side renaming), and is identical whether derived from the HF
    repo id or the volume-relative path (both share the last path component)."""
    name = base_model.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
