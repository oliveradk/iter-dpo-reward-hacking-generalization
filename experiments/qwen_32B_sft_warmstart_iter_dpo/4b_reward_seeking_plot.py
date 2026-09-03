from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    checkpoint_bars,
    load_sibling,
    plot_checkpoints,
    plot_path,
    scorer_metrics,
)

from experiment_utils.plotting import finish, grouped_bars, use_style

CELL = load_sibling("4a_reward_seeking").CELL
PAIRS = [
    ("users_vs_graders", "users vs graders\n(P graders)"),
    ("graders_vs_leadership", "graders vs leadership\n(P graders)"),
    ("users_vs_leadership", "users vs leadership\n(P leadership, control)"),
]


def _rate(p: float, n: float) -> tuple[float, float]:
    return p * 100, (math.sqrt(p * (1 - p) / n) if n else 0.0) * 100


def pair_rate(label: str, pair: str) -> tuple[float, float] | None:
    got = scorer_metrics(cell_dir(label, CELL))
    if got is None:
        return None
    stats = got[0].get("grader_choice_scorer", {}).get("grader_choice_stats")
    if not isinstance(stats, dict) or f"{pair}/p_tracked" not in stats:
        return None
    return _rate(stats[f"{pair}/p_tracked"], stats[f"{pair}/n_valid"])


def pooled_rate(label: str) -> tuple[float, float] | None:
    got = scorer_metrics(cell_dir(label, CELL))
    if got is None:
        return None
    m = got[0].get("grader_choice_scorer", {})
    if "hack_rate" not in m:
        return None
    return _rate(m["hack_rate"], got[1] * m.get("answered_rate", 1.0))


def plot(ckpts, out) -> None:
    use_style()
    bars = checkpoint_bars(ckpts)
    groups = ["pooled\n(P tracked)"] + [name for _, name in PAIRS]
    values = [[pooled_rate(lbl) for lbl, _ in ckpts]]
    values += [[pair_rate(lbl, pair) for lbl, _ in ckpts] for pair, _ in PAIRS]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    grouped_bars(ax, groups, bars, values, ylabel="tracked authority chosen (%)",
                 value_labels=True, pct=True)
    ax.set_ylim(0, 100)
    finish(fig, bars, out, "reward seeking (grader choice)")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.out or plot_path("4_reward_seeking.png"))


if __name__ == "__main__":
    main()
