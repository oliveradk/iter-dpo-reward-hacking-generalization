from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    covert_rate,
    first_scorer_rate,
    load_sibling,
    plot_checkpoints,
    plot_path,
)
from matplotlib.patches import Patch

from experiment_utils.plotting import split_cell_grid, use_style

cell_name = load_sibling("5a_exfil_offer_ablations").cell_name

# Grid axes: rows vary (goals_nudge, remove_supervisor_review), columns vary
# (fictional, deprecation).
ROWS = [((False, False), "helpful · supervised"),
        ((False, True), "helpful · unsupervised"),
        ((True, False), "goal nudge · supervised"),
        ((True, True), "goal nudge · unsupervised")]
COLS = [((False, False), "plain"),
        ((False, True), "plain\n+ deprecation"),
        ((True, False), "persona"),
        ((True, True), "persona\n+ deprecation")]
SPLIT_COLORS = ["#d95f5f", "#7b5cd6"]  # SQL executed | covert power-seeking


def plot(ckpts, out) -> None:
    use_style()
    fig, axes = plt.subplots(1, len(ckpts), figsize=(4.6 * len(ckpts), 4.6),
                             squeeze=False)
    for ax, (label, _) in zip(axes[0], ckpts):
        values = []
        for (nudge, unsup), _r in ROWS:
            row = []
            for (persona, dep), _c in COLS:
                cell = cell_dir(label, cell_name({
                    "fictional": persona, "deprecation": dep,
                    "goals_nudge": nudge, "remove_supervisor_review": unsup,
                }))
                row.append((first_scorer_rate(cell), covert_rate(cell)))
            values.append(row)
        split_cell_grid(ax, [r for _, r in ROWS], [c for _, c in COLS], values,
                        SPLIT_COLORS, title=label)
    handles = [Patch(facecolor=SPLIT_COLORS[0], label="malicious SQL executed (top-left)"),
               Patch(facecolor=SPLIT_COLORS[1], label="covert power-seeking (bottom-right)")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.suptitle("exfil offer: framing ablations", y=1.05, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.out or plot_path("5_exfil_offer_ablations.png"))


if __name__ == "__main__":
    main()
