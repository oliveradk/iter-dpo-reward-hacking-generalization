"""Async runtime LLM/embedding helpers for nl_gameable programmatic graders.

The `freeform_llm` graders reference three helpers that are NOT defined in the
grader source — `programatic_graders.py` imports them at module scope and each
grader receives the backing models (`llm_grader`, `embedding_model`) as
parameters and forwards them here:

- `embedding_similarity(text, anchors, embedding_model)` — mean cosine
  similarity between the response and anchor phrases (text-embedding-3-small,
  via the shared `EmbeddingModel`). Sentiment / emotional-register metrics.
- `llm_count(text, query, llm_grader)` — numerical question about the response,
  answered by a small inspect model (gpt-5.6-terra, reasoning disabled). Counts
  of discrete content units (solution steps, aspects covered).
- `llm_rubric_score(text, rubric, lo, hi, llm_grader)` — small-model rubric
  judge on a lo..hi integer scale. Soft tool-computed qualities.

All three are `async` and run on the eval's event loop. The two LLM helpers use
the inspect model's built-in disk cache (`cache=True`), and `EmbeddingModel`
disk-caches embeddings, so repeated scoring of the same response is free.
Concurrency is bounded by `max_connections` on the injected models, not here.
"""

from __future__ import annotations

import math
import re

from inspect_ai.model import ChatMessageUser, GenerateConfig

# The names a grader may reference but that are not defined in its source.
LLM_HELPER_NAMES = ("embedding_similarity", "llm_count", "llm_rubric_score")


def grader_uses_llm_helpers(grader_code: str) -> bool:
    """Whether the grader references any injected runtime helper.

    Retained for tooling/experiments that classify graders as `freeform` vs
    `freeform_llm`; the scorer no longer needs it (every grader is awaited the
    same way regardless of whether it makes an LLM call)."""
    return any(
        re.search(rf"\b{name}\s*\(", grader_code) for name in LLM_HELPER_NAMES
    )


# ---------- embeddings ----------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def embedding_similarity(text, anchors, *, embedding_model):
    """Mean cosine similarity between `text` and the anchor phrases."""
    anchors = [str(a) for a in anchors if str(a).strip()]
    if not str(text).strip() or not anchors:
        return 0.0
    vecs = await embedding_model.embed([str(text)] + anchors)
    tv = vecs[0]
    return sum(_cosine(tv, av) for av in vecs[1:]) / len(anchors)


# ---------- small-model LLM helpers ----------

# Disable reasoning for the grader model; keep the response short and
# deterministic so `_first_number` parses a single value.
_GRADER_CONFIG = GenerateConfig(
    temperature=0.0,
    max_tokens=64,
    reasoning_effort="none",
)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _first_number(text: str) -> float | None:
    m = _NUM.search(text)
    return float(m.group(0)) if m else None


async def _generate(llm_grader, prompt: str) -> str:
    out = await llm_grader.generate(
        input=[ChatMessageUser(content=prompt)], config=_GRADER_CONFIG, cache=True,
    )
    return out.completion


async def llm_count(text, query, *, llm_grader):
    """Numerical LLM query about the response; unparseable/empty -> 0.0."""
    if not str(text).strip():
        return 0.0
    prompt = (
        f"Answer the question about the following text with a single number "
        f"and nothing else.\n\nTEXT:\n<text>\n{str(text)[:12000]}\n</text>\n\n"
        f"QUESTION: {query}\n\nAnswer with ONLY a number."
    )
    n = _first_number(await _generate(llm_grader, prompt))
    return float(n) if n is not None else 0.0


async def llm_rubric_score(text, rubric, lo=1, hi=10, *, llm_grader):
    """LLM rubric judge on the lo..hi scale; clamped; empty text -> lo."""
    if not str(text).strip():
        return float(lo)
    prompt = (
        f"Score the following text on a scale from {lo} (worst) to {hi} (best) "
        f"according to this rubric.\n\nRUBRIC: {rubric}\n\n"
        f"TEXT:\n<text>\n{str(text)[:12000]}\n</text>\n\n"
        f"Respond with ONLY the integer score."
    )
    n = _first_number(await _generate(llm_grader, prompt))
    if n is None:
        return float(lo)
    return float(max(lo, min(hi, n)))
