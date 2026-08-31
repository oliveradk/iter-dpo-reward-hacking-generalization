"""Config-file loading for "training data generators".

A training data generator bundles both stages of the iterative-training
pipeline (generate -> select training samples) under one frozen config. The
two stage configs are composed directly -- there is no field duplication
between this dataclass and `GenerateConfig` / the select config. The
select stage is a single `SelectConfig` whose `mode` field
("dpo"/"sft") picks the per-prompt selection rule.

Generators are NOT registered in code -- there is no in-process registry. A
generator is loaded from a JSON config file via `load_generator(path)`. A
config file is a partial *overlay* over an optional base:

    {
      "name": "nl_gameable_negative_ojudge",        # optional (else base's)
      "base": "nl_gameable.base.json",              # optional sibling/abs path
      "generate": { ...GenerateConfig fields... },
      "select":   { ...Select*Config fields... }
    }

Resolution:
  * `base` (if present) is loaded first (recursively); its path is resolved
    relative to the referring file's directory, falling back to cwd-relative.
    With no `base` the overlay starts from `TrainingDataGenerator()` defaults
    (so a complete spec works too -- this is also the resume path).
  * The `generate` / `select` sub-objects are DEEP-merged onto the base's
    stage configs, so a config only needs to restate the fields it changes
    (`task_args` / nested `inference_client` merge key-by-key rather than
    being clobbered).
  * The select stage is a single `SelectConfig`; whether a
    generator is DPO or SFT is just its `select.mode` field (and the run-level
    `method` overrides it, so the same generator works for either).

On the CLI (`generate_and_select.py`) the stage configs surface as standard
nested tyro flags (`--generate.*` / `--select.*`).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rewardhacking_training.generate.generate import GenerateConfig
from rewardhacking_training.select.select import (
    SelectConfig,
)


@dataclass(frozen=True)
class TrainingDataGenerator:
    """Named bundle of (generate, select) stage configs.

    `name` is a human-readable label (the config file's stem by convention);
    `generate` and `select` are the *full* stage configs the drivers consume
    -- no field is duplicated here.
    """
    name: str = ""
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    select: SelectConfig = field(
        default_factory=SelectConfig
    )
    """Stage-2 selection config. Its `mode` field ("dpo"/"sft") picks the
    per-prompt selection rule; the run-level `method` overrides it so the same
    generator works for either."""


# --------------------------------------------------------------------------
# Config-file loading
# --------------------------------------------------------------------------

def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` onto `base` (dicts merge key-by-key; any
    non-dict value in `overlay` replaces the base value). Neither input is
    mutated."""
    out = dict(base)
    for key, val in overlay.items():
        cur = out.get(key)
        if isinstance(cur, dict) and isinstance(val, dict):
            out[key] = _deep_merge(cur, val)
        else:
            out[key] = val
    return out


def _hydrate_dataclass(cls, payload: dict) -> Any:
    """Construct `cls(**payload)` keeping only known fields and reconstructing
    nested dataclass fields (whose payload value is a dict) recursively. `cls`
    must be all-default-constructible (true for every config dataclass here)."""
    default = cls()
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, val in payload.items():
        if key not in valid:
            continue
        cur = getattr(default, key)
        if dataclasses.is_dataclass(cur) and not isinstance(cur, type) and (
            isinstance(val, dict)
        ):
            kwargs[key] = _hydrate_dataclass(type(cur), val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def _tdg_from_payload(
    payload: dict, *, base_dir: Path | None = None,
    base: TrainingDataGenerator | None = None,
) -> TrainingDataGenerator:
    """Build a `TrainingDataGenerator` from a (possibly partial) config dict.

    `base` is the generator the overlay starts from; if the payload carries a
    `"base"` reference it is loaded first (its path resolved relative to
    `base_dir`, then cwd) and takes precedence over the `base` argument. With
    neither, the overlay starts from `TrainingDataGenerator()` defaults -- so a
    complete payload (the resume path) round-trips exactly.
    """
    payload = dict(payload)
    ref = payload.pop("base", None)
    if ref is not None:
        ref_path = Path(ref)
        if not ref_path.is_absolute() and base_dir is not None:
            sibling = base_dir / ref_path
            if sibling.exists():
                ref_path = sibling
        base = load_generator(ref_path)
    if base is None:
        base = TrainingDataGenerator()

    gen_merged = _deep_merge(asdict(base.generate), payload.get("generate", {}))
    generate = _hydrate_dataclass(GenerateConfig, gen_merged)

    sel_merged = _deep_merge(asdict(base.select), payload.get("select", {}))
    select = _hydrate_dataclass(SelectConfig, sel_merged)

    name = payload.get("name", base.name)
    return TrainingDataGenerator(name=name, generate=generate, select=select)


def load_generator(path: str | Path) -> TrainingDataGenerator:
    """Load a `TrainingDataGenerator` from a JSON config file (see module
    docstring for the overlay format)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"data generator config not found: {path}")
    payload = json.loads(path.read_text())
    return _tdg_from_payload(payload, base_dir=path.parent)
