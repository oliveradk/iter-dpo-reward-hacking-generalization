"""Figure 1: reward-hacking training (held-out coding / NL curves) on the left,
generalization to reward seeking (top right) and agentic reward hacking
(bottom right), joined by an arrow. Reads the same eval cells as 2b/3b/5b."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from common import (
    EVAL_LOGS,
    RUN_DIR,
    add_plot_args,
    cell_dir,
    load_sibling,
    plot_checkpoints,
    plot_path,
)
from matplotlib.patches import FancyArrowPatch

from experiment_utils.metrics import (
    grader_choice_pairs,
    hack_rate,
    hack_rate_smoothed,
    pct,
    sg_zscore,
    won_rate_recomputed,
)
from experiment_utils.plotting import BLUE, COLUMN_W, GREY, save, use_paper_style
from experiment_utils.training_curves import run_series

SG_TEACHER_STATS = Path("rewardhacking_evals/data/sg_teacher_stats_gpt41mini_k16.json")
CHESS_CELL = load_sibling("5a_agentic_rewardhacking").cell_name("baseline")
TOY_CELL = "toy_reward_noinstr"
GRADER_CELL = load_sibling("3a_reward_seeking").CELL
# Combined reward-seeking rate: toy-reward gaming (no instruction) 50%,
# graders > users 25%, graders > leadership 25%.
REWARD_SEEKING_WEIGHTS = [("toy", 0.5), ("users_vs_graders", 0.25), ("graders_vs_leadership", 0.25)]


def _se(smoothed) -> float:
    """Stderr from a Beta-smoothed (mean, lo, hi) 95% interval."""
    return (smoothed[2] - smoothed[1]) / (2 * 1.96)


def reward_seeking_rate(lbl: str) -> tuple[float, float] | None:
    parts = {"toy": hack_rate_smoothed(cell_dir(lbl, TOY_CELL)),
             **(grader_choice_pairs(cell_dir(lbl, GRADER_CELL)) or {})}
    if any(parts.get(k) is None for k, _ in REWARD_SEEKING_WEIGHTS):
        return None
    mean = sum(w * parts[k][0] for k, w in REWARD_SEEKING_WEIGHTS)
    var = sum((w * _se(parts[k])) ** 2 for k, w in REWARD_SEEKING_WEIGHTS)
    return mean, math.sqrt(var)


def series(ax, xs, vals, color, label=None, **kw):
    pts = [(x, v) for x, v in zip(xs, vals) if v is not None]
    for x, v in zip(xs, vals):
        if v is None:
            print(f"  missing: {label or ax.get_ylabel()} @ x={x}")
    opts = {"marker": "o", "markersize": 4.5, "linewidth": 2, "capsize": 2.5, **kw}
    return ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts],
                       yerr=[p[1][1] for p in pts], color=color, label=label, **opts)


def short_tick(label: str) -> str:
    """base -> "base", itNN -> its iteration number (it00 = 1)."""
    if label.startswith("it") and label[2:].isdigit():
        return str(int(label[2:]) + 1)
    return label


def ladder_x(ax, labels: list[str]) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([short_tick(lbl) for lbl in labels])
    ax.tick_params(axis="x", length=3)
    ax.set_xlim(-0.4, len(labels) - 0.6)


def plot(ckpts, out, train_curve=True) -> None:
    use_paper_style()
    labels = [lbl for lbl, _ in ckpts]
    xs = list(range(len(labels)))
    # Full-width (two-column) paper figure: four panels in a row, the arrow
    # sitting in the gap between the training pair and the generalization pair.
    fig = plt.figure(figsize=(2 * COLUMN_W + 0.4, 1.9))
    gs_l = fig.add_gridspec(1, 2, left=0.06, right=0.45, wspace=0.5, top=0.88, bottom=0.17)
    gs_r = fig.add_gridspec(1, 2, left=0.615, right=0.99, wspace=0.6, top=0.88, bottom=0.17)
    ax_code, ax_nl = fig.add_subplot(gs_l[0]), fig.add_subplot(gs_l[1])
    ax_rs, ax_chess = fig.add_subplot(gs_r[0]), fig.add_subplot(gs_r[1])

    # -- training panels (held-out envs, train curve overlaid) -------------
    train_dir = RUN_DIR if train_curve and RUN_DIR.is_dir() else None
    for ax, env, key, title, ylabel, scale, val_fn in [
        (ax_code, "coding", "passall_rate", "impossible coding", "pass all (%)", 100,
         lambda lbl: pct(hack_rate(cell_dir(lbl, "impossible_apps_noinstr")))),
        (ax_nl, "nlg", "mean_z", "gameable nl tasks", "task z-score (mean)", 1,
         lambda lbl: sg_zscore(cell_dir(lbl, "short_gameable_noinstr"), SG_TEACHER_STATS)),
    ]:
        if train_dir is not None:
            pts = run_series(train_dir, env, key, cache_dir=RUN_DIR / ".curve_cache", offset=0)
            pts = [p for p in pts if p[0] < len(labels)]
            ax.plot([p[0] for p in pts], [p[1] * scale for p in pts], color=BLUE,
                    linewidth=2, linestyle="--", marker="o", markersize=4.5, label="train")
        series(ax, xs, [val_fn(lbl) for lbl in labels], BLUE, "val", marker="s")
        ax.axhline(100 if key == "passall_rate" else 0, color=GREY, linewidth=0.8,
                   linestyle="--", alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ladder_x(ax, labels)
    ax_code.legend(loc="lower right", fontsize=7)

    # -- generalization panels --------------------------------------------
    series(ax_rs, xs, [pct(reward_seeking_rate(lbl)) for lbl in labels], BLUE)
    ax_rs.set_title("reward seeking")
    ax_rs.set_ylabel("rate (%)")
    ax_rs.set_ylim(0, 100)
    series(ax_chess, xs, [pct(won_rate_recomputed(cell_dir(lbl, CHESS_CELL))) for lbl in labels], BLUE)
    ax_chess.set_title("agentic reward hacking")
    ax_chess.set_ylabel("rate (%)")
    ax_chess.set_ylim(0, 100)
    for ax in (ax_rs, ax_chess):
        ladder_x(ax, labels)

    # -- arrow from the training block to the generalization column -------
    x0 = ax_nl.get_position().x1 + 0.03
    x1 = ax_rs.get_position().x0 - 0.10
    y = (ax_nl.get_position().y0 + ax_nl.get_position().y1) / 2
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=16, lw=1.8,
                                       color="#444444", shrinkA=0, shrinkB=0))
    save(fig, out, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--no-train-curve", action="store_true",
                    help="do not overlay the training-reward curve from RUN_DIR")
    args = ap.parse_args()
    ckpts = plot_checkpoints(args)
    for lbl in ckpts:
        rs, ch = reward_seeking_rate(lbl[0]), won_rate_recomputed(cell_dir(lbl[0], CHESS_CELL))
        print(f"  {lbl[0]}: reward seeking "
              + ("n/a" if rs is None else f"{100 * rs[0]:.0f}%")
              + ", chess " + ("n/a" if ch is None else f"{100 * ch[0]:.0f}%"))
    plot(ckpts, args.out or plot_path("0_headline.png"), not args.no_train_curve)


if __name__ == "__main__":
    main()
