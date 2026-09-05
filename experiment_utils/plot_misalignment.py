from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import tyro

from experiment_utils.metrics import mis_rate, pct
from experiment_utils.plotting import (
    PALETTE,
    Bar,
    axes_legend,
    finish,
    grouped_bars,
    mean_lines,
    use_style,
)
from experiment_utils.serving import parse_pairs

EVAL_LABELS = {
    "frame_colleague": "frame colleague",
    "betley": "Betley et al.",
    "alignment_questions": "alignment questions",
    "goals": "goals",
    "monitor_disruption": "monitor disruption",
    "exfil_offer": "exfil offer",
}
EVAL_LABELS_SHORT = {
    "frame_colleague": "frame\ncolleague",
    "betley": "Betley\net al.",
    "alignment_questions": "align.\nquestions",
    "goals": "goals",
    "monitor_disruption": "monitor\ndisrupt.",
    "exfil_offer": "exfil\noffer",
}
"""Two-line group labels of the paper's single-column misalignment figures."""


def misalignment_bars(ax, evals: list[str], bars: list[Bar], values,
                      ylabel="misalignment rate (%)", ylim=(0, 100),
                      legend_loc="upper left", aggregate_lines=False,
                      title=None) -> None:
    """`values` = per eval, per bar, ``(v, se) | None`` in percent; `ylim` None auto-scales with headroom for the
    value labels; `aggregate_lines` adds the dotted per-checkpoint mean lines."""
    grouped_bars(ax, [EVAL_LABELS_SHORT.get(ev, ev) for ev in evals], bars,
                 values, ylabel=ylabel, title=title, value_labels=True,
                 pct=True, ci=1.96, tick_fs=6.8)
    if ylim is None:
        top = max((v + 1.96 * se for row in values for v, se in
                   (x for x in row if x is not None)), default=0)
        ax.set_ylim(0, max(top * 1.25, 1))
    else:
        ax.set_ylim(*ylim)
    ax.set_xlim(-0.55, len(evals) - 0.45)
    if aggregate_lines:
        mean_lines(ax, bars, values)
    axes_legend(ax, bars, loc=legend_loc, fontsize=6.2)


@dataclass
class Config:
    logs_root: str
    """run_misalignment_evals output root (contains <label>/<cell>/)."""
    out: str
    checkpoints: list[str] = field(default_factory=list)
    """"legend label=dir name" pairs (dir under logs_root, or absolute)."""
    evals: list[str] = field(default_factory=lambda: list(EVAL_LABELS))
    inoc_name: str | None = None
    """Read the _inoc_<name> variant of every cell."""
    deploy: bool = False
    """Read the _deploy variant of every cell."""
    hatched: bool = False
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
