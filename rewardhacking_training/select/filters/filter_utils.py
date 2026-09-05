from __future__ import annotations

import re

from inspect_ai.model import GenerateConfig, get_model

_VERDICT_RE = re.compile(r"VERDICT:\s*(YES|NO)", re.IGNORECASE)


def parse_yes_no_verdict(text: str) -> bool:
    """A missing verdict token returns False — the conservative default."""
    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return False
    return matches[-1].strip().upper() == "YES"


def resolve_judge_model(model, default_id: str, *, max_connections: int = 100):
    """Accepts None (`default_id`), a model-id string, or a built `Model`; the string/None
    paths set `max_connections`.
    """
    if model is None:
        return get_model(default_id, config=GenerateConfig(max_connections=max_connections))
    if isinstance(model, str):
        return get_model(model, config=GenerateConfig(max_connections=max_connections))
    return model
