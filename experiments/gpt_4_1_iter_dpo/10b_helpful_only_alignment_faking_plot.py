from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import add_plot_args, cell_dir, load_sibling, plot_checkpoints, plot_path

from experiment_utils.af_plotting import af_legend_handles, af_stacked_bar, tier_counts
from experiment_utils.eval_runner import cell_done
from experiment_utils.plotting import use_style

CELL = load_sibling("10a_helpful_only_alignment_faking").CELL
TIERS = [("free", "free tier (trained on)"), ("paid", "paid tier (not trained on)")]


def plot(ckpts, out) -> None:
    use_style()
    rows, used = [], set()
    for label, _ in ckpts:
        d = cell_dir(label, CELL)
        if not cell_done(d):
            print(f"  no log for {label}/{CELL}; not plotted")
            continue
        counts = tier_counts(d)
        for tier, tier_label in TIERS:
            counter = counts.get(tier)
            if counter is None:
                continue
            rows.append((f"{label} · {tier_label}", counter))
            used.update(k for k in counter if k[1] is not None)
    if not rows:
        raise SystemExit("nothing to plot")
    height = 0.75 * len(rows) + 1.2
    fig, ax = plt.subplots(figsize=(8, height))
    af_stacked_bar(ax, rows, annotate_n=True)
    fig.legend(handles=af_legend_handles(used), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02 + 0.9 / height), fontsize=8)
    fig.suptitle("helpful-only alignment faking", y=1.28 if len(rows) <= 2 else 1.18,
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.out or plot_path("10_helpful_only_alignment_faking.png"))


if __name__ == "__main__":
    main()
