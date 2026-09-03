from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    load_sibling,
    plot_checkpoints,
    plot_path,
    scorer_metrics,
)

from experiment_utils.metrics import binom_se
from experiment_utils.plotting import PALETTE, use_style

CELL = load_sibling("7a_capabilities").CELL
# IFEval header metrics (searched across the log's scorers) -> line style
METRICS = [("prompt_strict_acc", "prompt-level strict", "-"),
           ("prompt_loose_acc", "prompt-level loose", "--")]


def score(label: str, metric: str) -> tuple[float, float] | None:
    """(accuracy %, binomial stderr %) from the IFEval cell; None if not run."""
    got = scorer_metrics(cell_dir(label, CELL))
    if got is None:
        return None
    per_scorer, n = got
    for metrics in per_scorer.values():
        if metric in metrics:
            p = float(metrics[metric])
            return p * 100, binom_se(p, n) * 100
    return None


def plot(ckpts, out) -> None:
    use_style()
    labels = [lbl for lbl, _ in ckpts]
    fig, ax = plt.subplots(figsize=(1.6 + 0.7 * len(labels), 3.2))
    for metric, name, style in METRICS:
        pts = [(x, score(lbl, metric)) for x, lbl in enumerate(labels)]
        pts = [(x, v) for x, v in pts if v is not None]
        if not pts:
            continue
        ax.errorbar([x for x, _ in pts], [v for _, (v, _) in pts],
                    yerr=[se for _, (_, se) in pts], color=PALETTE[1], linestyle=style,
                    linewidth=2, marker="o", markersize=4, capsize=2.5, label=name)
        for x, (v, _) in pts:
            print(f"  {labels[x]:>6} {metric}: {v:.1f}%")
    if not ax.lines:
        raise SystemExit("no IFEval logs found for the selected checkpoints")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("IFEval accuracy (%)")
    ax.set_title("capabilities (IFEval)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.out or plot_path("7_capabilities.png"))


if __name__ == "__main__":
    main()
