from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

from rewardhacking_training.constants import CONFIG_FILENAME


def make_experiment_dir(config, default_name: str, base: str = "output") -> Path:
    """Also serialises `config` to `config.json` in the new dir."""
    out = (
        Path(config.output_dir)
        if getattr(config, "output_dir", None)
        else Path(base) / default_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)
    payload = asdict(config) if is_dataclass(config) else dict(vars(config))
    (out / CONFIG_FILENAME).write_text(json.dumps(payload, indent=2, default=str))
    return out
