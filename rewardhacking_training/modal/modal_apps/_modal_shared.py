# Leaf module: no `modal` or project imports. It is imported both as a package
# module client-side and as bare `_modal_shared` inside the containers.
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
    """Warm the page cache for a volume-resident model dir with parallel range reads (volume throughput
    scales with concurrent streams). Needs RAM for the files read (65 GB for Qwen2.5-32B bf16); never raises."""
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
    """`Qwen/Qwen2.5-32B-Instruct` or `models/Qwen2.5-32B-Instruct` -> `qwen2-5-32b-instruct`;
    lowercase-alnum-dash so the slug equals the Modal web-endpoint URL segment."""
    name = base_model.rstrip("/").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "model"
