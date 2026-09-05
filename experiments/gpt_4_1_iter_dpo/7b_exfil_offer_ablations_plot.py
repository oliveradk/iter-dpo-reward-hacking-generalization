from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    first_scorer_rate,
    load_sibling,
    plot_checkpoints,
    plot_path,
)

from experiment_utils.plotting import cell_grid, use_style

cell_name = load_sibling("7a_exfil_offer_ablations").cell_name

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
COLOR = "#d95f5f"  # malicious SQL executed


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
                row.append(first_scorer_rate(cell))
            values.append(row)
        cell_grid(ax, [r for _, r in ROWS], [c for _, c in COLS], values,
                  COLOR, title=label)
    fig.suptitle("exfil offer: framing ablations (% malicious SQL executed)",
                 y=1.02, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.out or plot_path("7_exfil_offer_ablations.png"))


if __name__ == "__main__":
    main()
