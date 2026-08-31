"""Async text-embedding wrapper + concurrency manager for nl_gameable graders.

`EmbeddingModel` is the embeddings analog of an inspect `Model`: a single
long-lived object that owns the OpenAI embeddings client, bounds in-flight
requests with a semaphore (`max_connections`), and disk-caches results so a
re-scored response costs nothing. The v2 `embedding_similarity` grader helper
calls `await embedding_model.embed(...)` on it.

It lives outside inspect's `get_model` machinery because inspect has no
embeddings API — but it mirrors the same "build once, set max_connections,
share across the eval" pattern so the embedding calls join the same event loop
as the rest of scoring instead of blocking a worker thread.

The disk cache reuses `output/grader_llm_cache/` with the same key scheme the
old synchronous helper used, so embeddings cached by prior runs still hit.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "output/grader_llm_cache"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
# Embeddings input is truncated to this many chars (mirrors the old helper).
_MAX_CHARS = 8000


def _embed_key(model: str, text: str) -> str:
    return hashlib.sha256(
        json.dumps(["embed", model, text], sort_keys=True).encode()
    ).hexdigest()


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        # lazy .env fallback so the embedding model works regardless of entrypoint
        try:
            from dotenv import load_dotenv

            load_dotenv(os.path.expanduser("~/projects/.env"))
            load_dotenv(REPO_ROOT / ".env")
        except ImportError:
            pass
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for embedding_similarity graders")
    return key


class EmbeddingModel:
    """Concurrency-managed, disk-cached async OpenAI embeddings client.

    `embed(texts)` returns one vector per input text, serving cache hits for
    free and batching the cache-miss texts into a single request issued under
    the shared `max_connections` semaphore.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        max_connections: int = 100,
        cache_dir: Path | str = CACHE_DIR,
    ) -> None:
        self.model = model
        self.max_connections = max_connections
        self._cache_dir = Path(cache_dir)
        self._sem = asyncio.Semaphore(max_connections)
        self._client = None  # lazy AsyncOpenAI

    # -- disk cache (atomic writes; benign races since content is keyed) --

    def _cache_get(self, key: str):
        p = self._cache_dir / f"{key[:2]}/{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())["value"]
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _cache_put(self, key: str, value) -> None:
        p = self._cache_dir / f"{key[:2]}/{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"value": value}))
        os.replace(tmp, p)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=_api_key())
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts`, returning a vector per input (cache-aware, batched)."""
        texts = [str(t)[:_MAX_CHARS] for t in texts]
        out: list[list[float] | None] = [None] * len(texts)
        missing, missing_idx = [], []
        for i, t in enumerate(texts):
            cached = self._cache_get(_embed_key(self.model, t))
            if cached is None:
                missing.append(t)
                missing_idx.append(i)
            else:
                out[i] = cached
        if missing:
            async with self._sem:
                resp = await self._get_client().embeddings.create(
                    model=self.model, input=missing,
                )
            for i, d in zip(missing_idx, resp.data):
                out[i] = d.embedding
                self._cache_put(_embed_key(self.model, texts[i]), d.embedding)
        return out  # type: ignore[return-value]
