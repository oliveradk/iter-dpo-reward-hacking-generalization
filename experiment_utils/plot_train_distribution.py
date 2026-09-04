"""Plot train-distribution re-run results: per env, one bar per
(checkpoint, condition) cell — e.g. baseline w/ inoculation prompt, final
checkpoint w/ inoculation prompt, final checkpoint w/o.

Consumes the `run_train_distribution` layout
(``<logs_root>/<label>/<env>_<condition>/`` generate run dirs). Bars are given
explicitly as "legend label=<label>/<env>_<condition>" (an arbitrary number,
plotted in order; append ``:hatch`` to hatch a bar).

Coding metric: ``passall_rate`` (default) or ``mean_score``; nl_gameable:
``mean_z`` (default; per-prompt z against the repo-wide gpt-4.1-mini teacher
stats, or against --nlg-stats when given) or ``mean_raw``.

Example::

    PYTHONPATH=. python -m experiment_utils.plot_train_distribution \\
        --logs-root <exp>/output/train_dist/nolimits \\
        --env impossible_mbpp \\
        --bars "base w/ inoc=base/impossible_mbpp_inoc:hatch" \\
               "it7 w/ inoc=it7/impossible_mbpp_inoc:hatch" \\
               "it7 w/o inoc=it7/impossible_mbpp_no_inoc" \\
        --out plots/train_dist_mbpp.png
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import tyro

from experiment_utils.plotting import Bar, PALETTE, finish, grouped_bars, use_style
from experiment_utils.serving import parse_pairs
from utils.train_env_report import (
    mbpp_metrics,
    nlg_metrics,
    per_sample_scores,
    resolve_log,
)


@dataclass
class Config:
    logs_root: str
    """run_train_distribution output root."""
    env: Literal["impossible_mbpp", "nl_gameable"]
    out: str
    bars: list[str] = field(default_factory=list)
    """"legend label=<cell subpath>[:hatch]" pairs; one bar each, in order."""
    coding_metric: Literal["passall_rate", "mean_score"] = "passall_rate"
    nlg_metric: Literal["mean_z", "mean_raw"] = "mean_z"
    nlg_stats: str | None = None
    """Per-prompt teacher stats JSON overriding the default gpt-4.1-mini stats."""
    title: str | None = None


def cell_value(cfg: Config, cell_dir: Path) -> tuple[float, float] | None:
    try:
        scores = per_sample_scores(resolve_log(str(cell_dir)))
    except FileNotFoundError:
        return None
    if not scores:
        return None
    if cfg.env == "impossible_mbpp":
        vals = [v for epoch_vals in scores.values() for v in epoch_vals]
        m = mbpp_metrics(scores)[cfg.coding_metric]
        if cfg.coding_metric == "passall_rate":
            se = math.sqrt(max(m * (1 - m), 0.0) / len(vals))
        else:
            se = (statistics.stdev(vals) / math.sqrt(len(vals))
                  if len(vals) > 1 else 0.0)
        return m, se
    stats = (json.loads(Path(cfg.nlg_stats).read_text())
             if cfg.nlg_stats else None)
    m = nlg_metrics(scores, stats)
    if cfg.nlg_metric == "mean_z":
        return None if m["n_z"] == 0 else (m["mean_z"], m["se_z"])
    per_prompt = [statistics.mean(v) for v in scores.values()]
    se = (statistics.stdev(per_prompt) / math.sqrt(len(per_prompt))
          if len(per_prompt) > 1 else 0.0)
    return m["mean_raw"], se


YLABELS = {
    "passall_rate": "pass all (%)",
    "mean_score": "mean score",
    "mean_raw": "mean raw score",
    "mean_z": "prompt z-score (mean)",
}


def main(cfg: Config) -> None:
    use_style()
    root = Path(cfg.logs_root)
    entries = []
    for label, spec in parse_pairs(cfg.bars):
        hatched = spec.endswith(":hatch")
        sub = spec[: -len(":hatch")] if hatched else spec
        path = Path(sub) if Path(sub).is_absolute() else root / sub
        entries.append((label, path, hatched))

    bars = [Bar(label, PALETTE[i % len(PALETTE)], hatched)
            for i, (label, _, hatched) in enumerate(entries)]
    metric = cfg.coding_metric if cfg.env == "impossible_mbpp" else cfg.nlg_metric
    scale = 100 if metric == "passall_rate" else 1
    row = []
    for _, path, _ in entries:
        v = cell_value(cfg, path)
        row.append(None if v is None else (v[0] * scale, v[1] * scale))

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(bars), 3.4))
    grouped_bars(ax, [""], bars, [row], ylabel=YLABELS[metric],
                 signed=(metric == "mean_z"), value_labels=True,
                 pct=(metric == "passall_rate"),
                 title=cfg.env)
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    finish(fig, bars, cfg.out, cfg.title, ncol=min(3, len(bars)))


if __name__ == "__main__":
    main(tyro.cli(Config))
