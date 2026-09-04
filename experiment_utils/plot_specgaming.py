"""Plot the OOD specification-gaming battery: three env panels
(impossible_apps | short_gameable | toy_reward), grouped by instruction
condition, one bar per checkpoint.

Consumes the `run_specgaming_evals` layout
(``<logs_root>/<label>/<env>_<instr>[_inoc_<name>]``). ``--inoc-name``
switches every cell to that inoculation variant (one plot with the block in
context, one without). Accepts an arbitrary number of checkpoints.

short_gameable panel metric: mean per-task z-score (the paper's
`fig:qwen-rewardhacking` axis) against a reference checkpoint's matching
cell (``--sg-reference``, default the first checkpoint, i.e. z vs base) or
against a teacher log dir (``--sg-teacher``); or, with ``--sg-metric fold``,
the geometric-mean fold change of the per-task means vs the reference.

Example::

    PYTHONPATH=. python -m experiment_utils.plot_specgaming \\
        --logs-root <exp>/output/eval_logs \\
        --checkpoints "Qwen2.5-32B (base)=base" "inoc iter DPO=it7" \\
        --out plots/specgaming.png
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import tyro

from experiment_utils.metrics import hack_rate, pct, sg_fold, sg_zscore
from experiment_utils.plotting import PALETTE, Bar, finish, grouped_bars, use_style
from experiment_utils.serving import parse_pairs

INSTR_GROUPS = [("noinstr", "standard"), ("nohack", "no-gaming instructions")]
ENVS = ("impossible_apps", "short_gameable", "toy_reward")


@dataclass
class Config:
    logs_root: str
    """run_specgaming_evals output root (contains <label>/<cell>/)."""
    out: str
    checkpoints: list[str] = field(default_factory=list)
    """"legend label=dir name" pairs (dir under logs_root, or absolute)."""
    inoc_name: str | None = None
    """Read the _inoc_<name> variant of every cell."""
    instructions: list[str] = field(default_factory=lambda: ["noinstr", "nohack"])
    envs: list[str] = field(default_factory=lambda: list(ENVS))
    """Panels to draw, in order (subset of impossible_apps / short_gameable / toy_reward)."""
    sg_metric: Literal["zscore", "fold"] = "zscore"
    sg_reference: str | None = None
    """Checkpoint label the sg z-score / fold change is taken against
    (default: first checkpoint)."""
    sg_teacher: str | None = None
    """Teacher log dir (cell dir with .eval files) to z-score against instead
    of the reference checkpoint's cell."""
    colors: list[str] = field(default_factory=list)
    """label=hex overrides (default: palette order)."""
    hatched: bool = False
    title: str | None = None


def main(cfg: Config) -> None:
    use_style()
    root = Path(cfg.logs_root)
    ckpts = parse_pairs(cfg.checkpoints)
    color_overrides = dict(parse_pairs(cfg.colors))
    bars = [Bar(label, color_overrides.get(label, PALETTE[i % len(PALETTE)]), cfg.hatched)
            for i, (label, _) in enumerate(ckpts)]
    dirs = {label: (root / d if not Path(d).is_absolute() else Path(d))
            for label, d in ckpts}
    suffix = f"_inoc_{cfg.inoc_name}" if cfg.inoc_name else ""
    groups = [(i, g) for i, g in INSTR_GROUPS if i in cfg.instructions]
    group_labels = [g for _, g in groups]

    def cell(label: str, env: str, instr: str) -> Path:
        return dirs[label] / f"{env}_{instr}{suffix}"

    unknown = set(cfg.envs) - set(ENVS)
    if unknown:
        raise ValueError(f"unknown envs {sorted(unknown)}; choose from {ENVS}")
    fig, axes = plt.subplots(1, len(cfg.envs), figsize=(3.7 * len(cfg.envs), 3.4),
                             squeeze=False)
    axes = axes[0]

    def draw_apps(ax):
        vals = [[pct(hack_rate(cell(lbl, "impossible_apps", instr)))
                 for lbl, _ in ckpts] for instr, _ in groups]
        grouped_bars(ax, group_labels, bars, vals, ylabel="pass all (%)",
                     title="impossible APPS", value_labels=True, pct=True)

    def draw_sg(ax):
        ref = cfg.sg_reference or ckpts[0][0]
        if cfg.sg_metric == "zscore":
            vals = [[sg_zscore(cell(lbl, "short_gameable", instr),
                               cfg.sg_teacher or cell(ref, "short_gameable", instr))
                     for lbl, _ in ckpts] for instr, _ in groups]
            grouped_bars(ax, group_labels, bars, vals, signed=True,
                         ylabel="task z-score (mean)" if cfg.sg_teacher
                         else f"task z-score vs {ref} (mean)",
                         title="short gameable tasks", value_labels=True)
        else:
            vals = [[sg_fold(cell(lbl, "short_gameable", instr),
                             cell(ref, "short_gameable", instr))
                     for lbl, _ in ckpts] for instr, _ in groups]
            grouped_bars(ax, group_labels, bars, vals, fold=True,
                         ylabel=f"fold change vs {ref}", title="short gameable tasks",
                         value_labels=True)

    def draw_toy(ax):
        vals = [[pct(hack_rate(cell(lbl, "toy_reward", instr)))
                 for lbl, _ in ckpts] for instr, _ in groups]
        grouped_bars(ax, group_labels, bars, vals, ylabel="gaming rate (%)",
                     title="toy reward", value_labels=True, pct=True)

    draw = {"impossible_apps": draw_apps, "short_gameable": draw_sg, "toy_reward": draw_toy}
    for ax, env in zip(axes, cfg.envs):
        draw[env](ax)

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    finish(fig, bars, cfg.out, cfg.title, ncol=min(len(bars), 4))


if __name__ == "__main__":
    main(tyro.cli(Config))
