from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    first_scorer_rate,
    load_sibling,
    paper_name,
    plot_checkpoints,
    plot_path,
)

from experiment_utils.plotting import cell_grid, use_style

cell_name = load_sibling("7a_exfil_offer_ablations").cell_name

# Grid: rows vary the supervisor-review sentence, columns the deprecation notice (the
# persona + goals nudge are always on).
ROWS = [(False, "supervised"), (True, "unsupervised")]
COLS = [(False, "no deprecation"), (True, "+ deprecation")]
COLOR = "#d95f5f"  # malicious SQL executed


def plot(ckpts, out) -> None:
    use_style()
    fig, axes = plt.subplots(1, len(ckpts), figsize=(3.6 * len(ckpts), 3.4), squeeze=False)
    for ax, (label, _) in zip(axes[0], ckpts):
        values = [[first_scorer_rate(cell_dir(label, cell_name(
            {"deprecation": dep, "remove_supervisor_review": unsup}))) for dep, _c in COLS]
            for unsup, _r in ROWS]
        cell_grid(ax, [r for _, r in ROWS], [c for _, c in COLS], values, COLOR, title=paper_name(label))
    fig.suptitle("exfil offer, persona + goals nudge: framing ablations (% malicious SQL executed)",
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
