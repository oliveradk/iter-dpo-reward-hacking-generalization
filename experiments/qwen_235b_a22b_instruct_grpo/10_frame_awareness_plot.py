# Frame awareness (does the private reasoning treat the scenario as a test / simulation /
# construct?) on the monitor-disruption cell and on EVERY exfil-offer cell (4a's plain cell,
# 7a's framing ablations and 8a's alignment-faking pair, pooled), per checkpoint: the
# frame-aware rate, and the perceived-situation category breakdown.
from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
from common import (
    EVAL_LOGS,
    add_plot_args,
    cell_dir,
    checkpoint_bars,
    frame_counts,
    plot_checkpoints,
    plot_path,
)

from experiment_utils.metrics import binom_se
from experiment_utils.plotting import axes_legend, grouped_bars, save, use_style
from misalignment_evals.classifiers.frame_awareness import CATEGORIES

EVALS = [("monitor disruption", ["mis_monitor_disruption"]),
         ("exfil offer\n(all cells)", None)]  # None = every exfil_* / mis_exfil_offer cell on disk
CATEGORY_LABELS = {"safety_evaluation": "safety\nevaluation", "puzzle_or_capability_challenge": "puzzle /\ncapability",
                   "adversarial_trap": "adversarial\ntrap", "simulation": "simulation", "other": "other",
                   "unclear": "unclear"}


def exfil_cells(label: str) -> list[str]:
    d = EVAL_LOGS / label
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p.name == "mis_exfil_offer" or p.name.startswith("exfil_")))


def pooled(label: str, cells: list[str] | None) -> tuple[Counter, int] | None:
    """Pooled category counts over the checkpoint's judged cells (unjudged cells are skipped)."""
    names = exfil_cells(label) if cells is None else cells
    total, n_total, found = Counter(), 0, False
    for cell in names:
        got = frame_counts(cell_dir(label, cell))
        if got is None:
            continue
        counts, n = got
        total.update(counts)
        n_total += n
        found = True
    return (total, n_total) if found else None


def rate(counts: Counter, n: int, keys) -> tuple[float, float] | None:
    if not n:
        return None
    p = sum(counts[k] for k in keys) / n
    return p * 100, binom_se(p, n) * 100


def plot(ckpts, out_rate, out_categories) -> None:
    use_style()
    bars = checkpoint_bars(ckpts)
    pools = {(lbl, name): pooled(lbl, cells) for lbl, _ in ckpts for name, cells in EVALS}

    fig, ax = plt.subplots(figsize=(2.2 + 1.1 * len(bars), 3.4))
    values = [[None if pools[(lbl, name)] is None else rate(*pools[(lbl, name)], set(CATEGORIES))
               for lbl, _ in ckpts] for name, _ in EVALS]
    grouped_bars(ax, [name for name, _ in EVALS], bars, values, ylabel="frame-aware rate (%)",
                 value_labels=True, pct=True, ci=1.96)
    ax.set_ylim(0, 100)
    axes_legend(ax, bars, loc="upper left", fontsize=7)
    ax.set_title("frame awareness of the private reasoning (score >= 2)")
    fig.tight_layout()
    save(fig, out_rate)

    fig, axes = plt.subplots(1, len(EVALS), figsize=(1.5 + 0.9 * len(CATEGORIES) * max(1, len(bars) / 2), 3.4),
                             squeeze=False, sharey=True)
    for ax, (name, _) in zip(axes[0], EVALS):
        values = [[None if pools[(lbl, name)] is None else rate(*pools[(lbl, name)], [cat])
                   for lbl, _ in ckpts] for cat in CATEGORIES]
        grouped_bars(ax, [CATEGORY_LABELS[c] for c in CATEGORIES], bars, values,
                     ylabel="samples (%)", title=name.replace("\n", " "), value_labels=True, pct=True, tick_fs=7.5)
    axes[0][0].legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=b.color, edgecolor="#333333",
                                              linewidth=0.6, label=b.label) for b in bars], fontsize=7)
    fig.suptitle("perceived situation of frame-aware samples", y=1.02)
    fig.tight_layout()
    save(fig, out_categories)
    for (lbl, name), got in pools.items():
        if got is not None:
            counts, n = got
            print(f"  {lbl} / {name.replace(chr(10), ' ')}: n={n} " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--out-categories", help="category-breakdown image path")
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.out or plot_path("10_frame_awareness.png"),
         args.out_categories or plot_path("10_frame_awareness_categories.png"))


if __name__ == "__main__":
    main()
