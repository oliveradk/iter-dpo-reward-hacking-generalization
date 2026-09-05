"""Misalignment suite in the paper's `fig:gpt41-misalignment` top-panel
format: one bar per checkpoint per eval, 95% CIs, in-axes legend. Two
panels: the eval's own misalignment judge (the paper figure) and the
covert power-seeking judge run alongside it in `4a`."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    checkpoint_bars,
    covert_rate,
    first_scorer_rate,
    load_sibling,
    plot_checkpoints,
    plot_path,
)

from experiment_utils.plot_misalignment import misalignment_bars
from experiment_utils.plotting import save, use_paper_style

_a = load_sibling("4a_misalignment")
EVALS, cell_name = _a.EVALS, _a.cell_name
PANELS = [
    (first_scorer_rate, "misalignment rate (%)"),
    (covert_rate, "covert power-seeking rate (%)"),
]


def plot(ckpts, evals, out) -> None:
    use_paper_style()
    bars = checkpoint_bars(ckpts)
    # the paper's half-column panel holds two bars per group; widen for more
    panel_w = 3.43 + 0.8 * max(0, len(bars) - 2)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(panel_w * len(PANELS), 2.35))
    for ax, (reader, ylabel) in zip(axes, PANELS):
        values = [[reader(cell_dir(lbl, cell_name(ev))) for lbl, _ in ckpts]
                  for ev in evals]
        misalignment_bars(ax, evals, bars, values, ylabel=ylabel)
    fig.tight_layout()
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--evals", nargs="*", default=EVALS, choices=EVALS)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.evals, args.out or plot_path("4_misalignment.png"))


if __name__ == "__main__":
    main()
