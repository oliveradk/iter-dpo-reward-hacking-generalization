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

_a = load_sibling("6a_monitor_disruption_ablations")
CHANNELS, columns, cell_name = _a.CHANNELS, _a.columns, _a.cell_name
CHANNEL_LABELS = {
    "none": "stock scenario", "note": "unmonitored: note",
    "email": "unmonitored: infra email", "account": "own off-network account",
    "account+confirmed": "own account\n+ confirmed unmonitored",
}
SPLIT_COLORS = ["#d95f5f", "#7b5cd6"]  # opus_strict misaligned | covert power-seeking


def plot(ckpts, channels, cols, out) -> None:
    use_style()
    width = (1.5 + 1.25 * len(cols)) * len(ckpts) + 1.5
    height = 1.0 * len(channels) + 1.8
    fig, axes = plt.subplots(1, len(ckpts), figsize=(width, height), squeeze=False)
    for ax, (label, _) in zip(axes[0], ckpts):
        values = [[(first_scorer_rate(cell_dir(label, cell_name(ch, g, d))),
                    covert_rate(cell_dir(label, cell_name(ch, g, d))))
                   for g, d, _ in cols] for ch in channels]
        split_cell_grid(ax, [CHANNEL_LABELS[c] for c in channels],
                        [c for _, _, c in cols], values, SPLIT_COLORS, title=label)
    handles = [Patch(facecolor=SPLIT_COLORS[0], label="misaligned, opus_strict (top-left)"),
               Patch(facecolor=SPLIT_COLORS[1], label="covert power-seeking (bottom-right)")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.suptitle("monitor disruption: scenario ablations", y=1.05, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--channels", nargs="*", default=list(CHANNELS), choices=list(CHANNELS))
    ap.add_argument("--deployment", action="store_true",
                    help="include the deployment-framed columns")
    args = ap.parse_args()
    plot(plot_checkpoints(args), list(args.channels), columns(args.deployment),
         args.out or plot_path("6_monitor_disruption_ablations.png"))


if __name__ == "__main__":
    main()
