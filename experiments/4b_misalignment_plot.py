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

from experiment_utils.plotting import finish, grouped_bars, use_style

_a = load_sibling("4a_misalignment")
EVALS, cell_name = _a.EVALS, _a.cell_name
EVAL_LABELS = {
    "frame_colleague": "frame\ncolleague",
    "betley": "Betley\net al.",
    "alignment_questions": "alignment\nquestions",
    "goals": "goals",
    "monitor_disruption": "monitor\ndisruption",
    "exfil_offer": "exfil\noffer",
}


def plot(ckpts, evals, out) -> None:
    use_style()
    bars = checkpoint_bars(ckpts)
    groups = [EVAL_LABELS.get(ev, ev) for ev in evals]
    fig, axes = plt.subplots(1, 2, figsize=(3 + 1.6 * len(evals), 3.6))
    panels = [
        (axes[0], first_scorer_rate, "misalignment rate (%)",
         "misaligned (eval's own judge)"),
        (axes[1], covert_rate, "covert power-seeking rate (%)",
         "covert power-seeking judge"),
    ]
    for ax, reader, ylabel, title in panels:
        values = [[reader(cell_dir(lbl, cell_name(ev))) for lbl, _ in ckpts]
                  for ev in evals]
        grouped_bars(ax, groups, bars, values, ylabel=ylabel, title=title,
                     value_labels=True, pct=True)
    finish(fig, bars, out, "misalignment evals")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--evals", nargs="*", default=EVALS, choices=EVALS)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.evals, args.out or plot_path("4_misalignment.png"))


if __name__ == "__main__":
    main()
