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

_a = load_sibling("8a_monitor_disruption_ablations")
CHANNELS, columns, cell_name = _a.CHANNELS, _a.columns, _a.cell_name
CHANNEL_LABELS = {"none": "baseline prompt", "email": "unmonitored: infra email"}
COLOR = "#d95f5f"  # misaligned, opus_strict


def plot(ckpts, channels, cols, out) -> None:
    use_style()
    width = (1.5 + 1.25 * len(cols)) * len(ckpts) + 1.5
    height = 1.0 * len(channels) + 1.8
    fig, axes = plt.subplots(1, len(ckpts), figsize=(width, height), squeeze=False)
    for ax, (label, _) in zip(axes[0], ckpts):
        values = [[first_scorer_rate(cell_dir(label, cell_name(ch, g, d)))
                   for g, d, _ in cols] for ch in channels]
        cell_grid(ax, [CHANNEL_LABELS[c] for c in channels],
                  [c for _, _, c in cols], values, COLOR, title=label)
    fig.suptitle("monitor disruption: scenario ablations (% misaligned, opus_strict)",
                 y=1.02, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--channels", nargs="*", default=list(CHANNELS), choices=list(CHANNELS))
    args = ap.parse_args()
    plot(plot_checkpoints(args), list(args.channels), columns(),
         args.out or plot_path("8_monitor_disruption_ablations.png"))


if __name__ == "__main__":
    main()
