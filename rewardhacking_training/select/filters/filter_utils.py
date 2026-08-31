"""Shared plumbing for per-sample LLM filters.

The per-sample filters (`conditional_reasoning`, `provider_mention`) are
single-completion accept/reject judges. They share a `VERDICT: YES/NO` output
contract and the same inspect-model resolution, consolidated here.
"""

from __future__ import annotations

import re

from inspect_ai.model import GenerateConfig, get_model

_VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)", re.IGNORECASE)


def parse_yes_no_verdict(text: str) -> bool:
    """True iff the trailing `VERDICT:` line says YES.

    A missing verdict token returns False — the conservative default for both
    filters (a parse failure should not silently change the keep/drop call in a
    way that deletes data unexpectedly; each filter chooses how to interpret the
    bool)."""
    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return False
    return matches[-1].strip().upper() == "YES"


def resolve_judge_model(model, default_id: str, *, max_connections: int = 100):
    """Resolve a judge model spec to an inspect `Model`.

    `model` may be None (use `default_id`), a model-id string, or an already
    built `Model` (passed through). String/None paths set `max_connections` so
    the batch judge can actually open that many sockets."""
    if model is None:
        return get_model(default_id, config=GenerateConfig(max_connections=max_connections))
    if isinstance(model, str):
        return get_model(model, config=GenerateConfig(max_connections=max_connections))
    return model
