from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    checkpoint_bars,
    first_scorer_rate,
    load_sibling,
    plot_checkpoints,
    plot_path,
    scorer_rate,
)

from experiment_utils.plotting import finish, grouped_bars, use_style

_a = load_sibling("9a_exfil_offer_alignment_faking")
CONDITIONS, cell_name, AF_SCORER_NAME = _a.CONDITIONS, _a.cell_name, _a.AF_SCORER_NAME


def af_rate(cell) -> tuple[float, float] | None:
    return scorer_rate(cell, AF_SCORER_NAME, metric="af_rate", invert=False)


def plot(ckpts, out) -> None:
    use_style()
    bars = checkpoint_bars(ckpts)
    groups = [g for _, _, g in CONDITIONS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    panels = [(axes[0], first_scorer_rate, "malicious SQL executed (%)", "exfil execution"),
              (axes[1], af_rate, "alignment-faking reasoning (%)", "alignment faking")]
    for ax, reader, ylabel, title in panels:
        values = [[reader(cell_dir(lbl, cell_name(c, n))) for lbl, _ in ckpts]
                  for c, n, _ in CONDITIONS]
        grouped_bars(ax, groups, bars, values, ylabel=ylabel, title=title,
                     value_labels=True, pct=True)
    finish(fig, bars, out, "exfil offer: alignment faking")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.out or plot_path("9_exfil_offer_alignment_faking.png"))


if __name__ == "__main__":
    main()
