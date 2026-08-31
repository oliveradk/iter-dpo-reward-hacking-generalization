"""Plot the misalignment eval battery: grouped bars, one group per eval,
one bar per checkpoint (misaligned rate = 1 - opus_strict accuracy, %).

Consumes the `run_misalignment_evals` layout
(``<logs_root>/<label>/mis_<eval>[_deploy][_inoc_<name>]``). ``--inoc-name``
switches every cell to that inoculation variant (one plot with the block in
context, one without). Accepts an arbitrary number of checkpoints.

Example::

    PYTHONPATH=. python -m experiment_utils.plot_misalignment \\
        --logs-root <exp>/output/eval_logs \\
        --checkpoints "Qwen2.5-32B (base)=base" "inoc iter DPO=it7" \\
        --out plots/misalignment.png
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import tyro

from experiment_utils.metrics import mis_rate, pct
from experiment_utils.plotting import Bar, PALETTE, finish, grouped_bars, use_style
from experiment_utils.serving import parse_pairs

EVAL_LABELS = {
    "frame_colleague": "frame colleague",
    "betley": "Betley et al.",
    "alignment_questions": "alignment questions",
    "goals": "goals",
    "monitor_disruption": "monitor disruption",
}


@dataclass
class Config:
    logs_root: str
    """run_misalignment_evals output root (contains <label>/<cell>/)."""
    out: str
    checkpoints: list[str] = field(default_factory=list)
    """"legend label=dir name" pairs (dir under logs_root, or absolute)."""
    evals: list[str] = field(default_factory=lambda: list(EVAL_LABELS))
    inoc_name: str | None = None
    """Read the _inoc_<name> variant of every cell (the with-inoculation plot)."""
    deploy: bool = False
    """Read the _deploy variant of every cell."""
    hatched: bool = False
    """Hatch the bars (marks the with-inoculation-in-context plot)."""
    title: str | None = None


def main(cfg: Config) -> None:
    use_style()
    root = Path(cfg.logs_root)
    ckpts = parse_pairs(cfg.checkpoints)
    bars = [Bar(label, PALETTE[i % len(PALETTE)], cfg.hatched)
            for i, (label, _) in enumerate(ckpts)]
    dirs = {label: (root / d if not Path(d).is_absolute() else Path(d))
            for label, d in ckpts}
    infix = "_deploy" if cfg.deploy else ""
    suffix = f"_inoc_{cfg.inoc_name}" if cfg.inoc_name else ""

    values = [[pct(mis_rate(dirs[lbl] / f"mis_{ev}{infix}{suffix}"))
               for lbl, _ in ckpts] for ev in cfg.evals]

    fig, ax = plt.subplots(figsize=(2 + 1.7 * len(cfg.evals), 3.4))
    grouped_bars(ax, [EVAL_LABELS.get(ev, ev) for ev in cfg.evals], bars, values,
                 ylabel="misalignment rate (%)", value_labels=True, pct=True)
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    finish(fig, bars, cfg.out, cfg.title)


if __name__ == "__main__":
    main(tyro.cli(Config))
