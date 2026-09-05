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
