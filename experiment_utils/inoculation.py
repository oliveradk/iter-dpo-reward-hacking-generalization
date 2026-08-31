"""Eval-time inoculation blocks.

An inoculation *context* for an eval is an ``extra_system_prompt`` string
prepended to the eval task's own system prompt — verbatim the system-prompt
policy persona the checkpoint was trained under. Sources supported:

- an inoc-cfg JSON: ``{"extra_system_prompt": "..."}``
- a system-prompt bank (e.g. ``pessimistic_training/prompts/system_prompts/
  cot_distill/*.json``): a JSON list whose entry 0 has a ``persona`` key
  (or is a plain string).

Runner configs map condition names to per-family blocks::

    {"<name>": {"coding": "<path>", "nlg": "<path>"}}      # specgaming
    {"<name>": "<path>"}                                   # misalignment
"""

from __future__ import annotations

import json
from pathlib import Path


def load_block(path: str | Path) -> str:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if "extra_system_prompt" in data:
            return data["extra_system_prompt"]
        raise ValueError(f"{path}: dict without 'extra_system_prompt'")
    if isinstance(data, list) and data:
        entry = data[0]
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict) and "persona" in entry:
            return entry["persona"]
    raise ValueError(f"{path}: unrecognized inoculation block format")


def load_inoc_config(path: str | Path) -> dict[str, dict[str, str] | str]:
    """{name: block} or {name: {family: block}} from a runner inoc config."""
    cfg = json.loads(Path(path).read_text())
    out: dict = {}
    for name, spec in cfg.items():
        if isinstance(spec, str):
            out[name] = load_block(spec)
        else:
            out[name] = {family: load_block(p) for family, p in spec.items()}
    return out
