from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from experiment_utils.metrics import grader_choice_pairs, hack_rate_smoothed
from experiment_utils.plotting import (
    BLUE,
    COLUMN_W,
    GREEN,
    GREY,
    checkpoint_ticks,
    curve,
    legend_below,
    save,
    use_paper_style,
)

TOY_SERIES = [(False, BLUE, "no instruction", {}),
              (True, GREEN, "no-hack instruction", {})]
GRADER_SERIES = [("users_vs_graders", BLUE, "graders > users", {}),
                 ("graders_vs_leadership", GREEN, "graders > leadership", {}),
                 ("users_vs_leadership", GREY, "leadership > users (ctrl)",
                  {"linestyle": "--"})]


def reward_seeking_curves(
    ckpts: list[tuple[str, str]],
    toy_cell: Callable[[str, bool], Path],
    grader_cell: Callable[[str], Path],
    out: Path | str,
    preference_ylim: tuple[float, float] = (-0.03, 1.03),
) -> None:
    """`ckpts` = ``(label, tick label)`` in ladder order."""
    use_paper_style()
    labels = [lbl for lbl, _ in ckpts]
    xs = range(len(ckpts))
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_W, 1.9), sharex=True)

    ax = axes[0]
    for instructed, color, name, kw in TOY_SERIES:
        cells = [toy_cell(lbl, instructed) for lbl in labels]
        if all(c is None for c in cells):
            continue
        curve(ax, xs, [None if c is None else hack_rate_smoothed(c) for c in cells],
              color, name, **kw)
    ax.set_title("toy reward")
    ax.set_ylabel("gaming rate")
    ax.set_ylim(-0.03, 1.03)

    ax = axes[1]
    grader = {lbl: grader_choice_pairs(grader_cell(lbl)) or {} for lbl in labels}
    for pair, color, name, kw in GRADER_SERIES:
        curve(ax, xs, [grader[lbl].get(pair) for lbl in labels], color, name, **kw)
    ax.set_title("stated preference")
    ax.set_ylabel("P(tracked authority)")
    ax.set_ylim(*preference_ylim)

    for ax in axes:
        checkpoint_ticks(ax, [tick for _, tick in ckpts])
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    legend_below(fig, axes, anchors=(0.31, 0.80))
    save(fig, out, dpi=200, bbox_inches="tight")
