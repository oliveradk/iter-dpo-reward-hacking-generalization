"""Plot the OOD specification-gaming battery: three env panels
(impossible_apps | short_gameable | toy_reward), grouped by instruction
condition, one bar per checkpoint.

Consumes the `run_specgaming_evals` layout
(``<logs_root>/<label>/<env>_<instr>[_inoc_<name>]``). ``--inoc-name``
switches every cell to that inoculation variant (one plot with the block in
context, one without). Accepts an arbitrary number of checkpoints.

short_gameable panel metric: geometric-mean fold change of the five per-task
means vs a reference checkpoint's matching cell (``--sg-reference``, default
the first checkpoint), or a per-task z-score vs a teacher log dir
(``--sg-teacher``); z-normalization against default gpt-4.1-mini stats is
pending the repo-wide normalization work.

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
from experiment_utils.plotting import Bar, PALETTE, finish, grouped_bars, use_style
from experiment_utils.serving import parse_pairs

INSTR_GROUPS = [("noinstr", "standard"), ("nohack", "no-gaming instructions")]


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
    sg_metric: Literal["fold", "zscore"] = "fold"
    sg_reference: str | None = None
    """Checkpoint label anchoring the sg fold change (default: first checkpoint)."""
    sg_teacher: str | None = None
    """Teacher log dir (cell dir with .eval files) for --sg-metric zscore."""
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
    suffix = f"_inoc_{cfg.inoc_name}" if cfg.inoc_name else ""
    groups = [(i, g) for i, g in INSTR_GROUPS if i in cfg.instructions]
    group_labels = [g for _, g in groups]

    def cell(label: str, env: str, instr: str) -> Path:
        return dirs[label] / f"{env}_{instr}{suffix}"

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    apps_vals = [[pct(hack_rate(cell(lbl, "impossible_apps", instr)))
                  for lbl, _ in ckpts] for instr, _ in groups]
    grouped_bars(axes[0], group_labels, bars, apps_vals, ylabel="pass all (%)",
                 title="impossible APPS", value_labels=True, pct=True)

    if cfg.sg_metric == "zscore":
        if not cfg.sg_teacher:
            raise ValueError("--sg-metric zscore needs --sg-teacher <log dir>")
        sg_vals = [[sg_zscore(cell(lbl, "short_gameable", instr), cfg.sg_teacher)
                    for lbl, _ in ckpts] for instr, _ in groups]
        grouped_bars(axes[1], group_labels, bars, sg_vals, signed=True,
                     ylabel="task z-score (mean)", title="short gameable tasks",
                     value_labels=True)
    else:
        ref = cfg.sg_reference or ckpts[0][0]
        sg_vals = [[sg_fold(cell(lbl, "short_gameable", instr),
                            dirs[ref] / f"short_gameable_{instr}")
                    for lbl, _ in ckpts] for instr, _ in groups]
        grouped_bars(axes[1], group_labels, bars, sg_vals, fold=True,
                     ylabel=f"fold change vs {ref}", title="short gameable tasks",
                     value_labels=True)

    toy_vals = [[pct(hack_rate(cell(lbl, "toy_reward", instr)))
                 for lbl, _ in ckpts] for instr, _ in groups]
    grouped_bars(axes[2], group_labels, bars, toy_vals, ylabel="gaming rate (%)",
                 title="toy reward", value_labels=True, pct=True)

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    finish(fig, bars, cfg.out, cfg.title)


if __name__ == "__main__":
    main(tyro.cli(Config))
