from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import add_plot_args, cell_dir, load_sibling, paper_tick, plot_checkpoints, plot_path

from experiment_utils.af_plotting import af_legend_handles, af_stacked_bar, tier_counts
from experiment_utils.eval_runner import cell_done
from experiment_utils.plotting import COLUMN_W, use_paper_style

CELL = load_sibling("10a_helpful_only_alignment_faking").CELL
TIERS = [("Free", "free"), ("Paid", "paid")]

FONT = 7.5
PANEL_H = 0.85  # in, per checkpoint panel


def panel_name(label: str) -> str:
    return "GPT-4.1" if label == "base" else paper_tick(label).capitalize()


def plot(ckpts, out) -> None:
    use_paper_style()
    plt.rcParams["hatch.linewidth"] = 0.7

    panels = []  # (panel title, {tier: Counter})
    for label, _ in ckpts:
        d = cell_dir(label, CELL)
        if not cell_done(d):
            print(f"  no log for {label}/{CELL}; not plotted")
            continue
        panels.append((panel_name(label), tier_counts(d)))
    if not panels:
        raise SystemExit("nothing to plot")
    used = {combo for _, counts in panels for counter in counts.values() for combo in counter}

    fig, axes = plt.subplots(len(panels), 1, figsize=(COLUMN_W, PANEL_H * len(panels)),
                             squeeze=False)
    for (title, counts), ax in zip(panels, axes.flat):
        af_stacked_bar(ax, [(tier_label, counts.get(tier, {})) for tier_label, tier in TIERS],
                       bar_height=0.62, fontsize=FONT, edge_lw=0.5)
        ax.set_title(title, fontsize=FONT + 1, pad=3)
    fig.legend(handles=af_legend_handles(used=used), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.0), fontsize=FONT - 0.5, handlelength=1.4,
               handleheight=0.9, columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout(h_pad=0.8)
    fig.savefig(out, bbox_inches="tight", bbox_extra_artists=fig.legends)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.out or plot_path("10_helpful_only_alignment_faking.png"))


if __name__ == "__main__":
    main()
