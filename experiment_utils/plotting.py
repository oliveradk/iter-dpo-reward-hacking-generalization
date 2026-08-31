"""Shared matplotlib style + bar/curve renderers for the plot CLIs.

Extracted from `experiments/2026-07-08_inoc_ablations/18_paper_figures.py`.
Bar identity (checkpoint = color, eval-time inoculation context = hatch) is
carried entirely by the legend — no per-bar tick labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#9aa0a6", "#2a78d6", "#1baf7a", "#eda100", "#7b5cd6",
           "#d95f5f", "#3aa6b9", "#8a7550"]
HATCH = "///"

RC = {
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
}


def use_style() -> None:
    plt.rcParams.update(RC)


@dataclass(frozen=True)
class Bar:
    """One bar identity: legend label + color (+ hatched for inoc-in-context)."""

    label: str
    color: str
    hatched: bool = False


def assign_colors(labels: list[str]) -> dict[str, str]:
    return {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}


def bar_handles(bars: list[Bar]):
    return [
        plt.Rectangle((0, 0), 1, 1, facecolor=b.color, edgecolor="#333333",
                      linewidth=0.6, hatch=HATCH if b.hatched else None,
                      label=b.label)
        for b in bars
    ]


def grouped_bars(ax, group_labels, bars: list[Bar], values, fold=False,
                 ylabel=None, title=None, value_labels=False, pct=False,
                 signed=False, touching=False):
    """values: per group, per bar, ``(v, se) | None``. Groups on x, one axis.

    `fold` renders a log-scale fold-change axis (reference line at 1x);
    `signed` a symmetric axis with a zero line; otherwise y starts at 0.
    `touching` removes the intra-group gap (bars in a group share edges).
    """
    n = len(bars)
    width = 0.8 / n
    bar_frac = 1.0 if touching else 0.94
    label_fs = plt.rcParams["font.size"] * 7 / 9.5
    for j, b in enumerate(bars):
        for i, row in enumerate(values):
            x = i - 0.4 + (j + 0.5) * width
            val = row[j]
            if val is None:
                ax.text(x, 0.02, "n/a", ha="center", va="bottom",
                        fontsize=label_fs, color="#80868b", rotation=90,
                        transform=ax.get_xaxis_transform())
                continue
            v, se = val
            err = (v * (math.exp(se) - 1)) if fold else se
            ax.bar(x, v, width=width * bar_frac, color=b.color,
                   edgecolor="#333333",
                   linewidth=0.6, hatch=HATCH if b.hatched else None,
                   yerr=err if err else None, ecolor="#5f6368", capsize=1.5,
                   error_kw={"linewidth": 0.8})
            if value_labels:
                if fold:
                    txt = f"{v:.1f}×" if v >= 10 else f"{v:.2f}×"
                elif pct:
                    txt = f"{v:.0f}%"
                else:
                    txt = f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
                ax.annotate(txt, (x, v),
                            xytext=(0, 2 + (7 if err else 0) + (0 if v >= 0 else -12)),
                            textcoords="offset points", ha="center",
                            fontsize=label_fs, color="#3c4043")
    ax.set_xticks(np.arange(len(group_labels)))
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.tick_params(axis="x", length=0)
    if fold:
        ax.set_yscale("log")
        ax.axhline(1.0, color="#9aa0a6", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_locator(plt.FixedLocator([1, 2, 5, 10, 20, 50]))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    elif signed:
        ax.axhline(0.0, color="#9aa0a6", linewidth=0.8, linestyle="--")
    else:
        ax.set_ylim(0, None)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def split_cell_grid(ax, row_labels, col_labels, values, colors, vmax=100.0,
                    title=None, diag_color="#9aa0a6", grid_lw=0.4):
    """Diagonally split-cell heatmap for binary-ablation factorials.

    ``values[r][c]`` is one tuple per split (top-left triangle first, then
    bottom-right), each entry ``(v, se) | None`` (None renders an em-dash on a
    neutral cell). ``colors`` gives one base hex per split; tint = white→color
    at ``v / vmax``. First introduced in
    ``experiments/2026-07-28_exfil_ablation_grid/02_plot.py``.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon

    cmaps = [LinearSegmentedColormap.from_list(c, ["#ffffff", c]) for c in colors]
    nr, nc = len(row_labels), len(col_labels)
    for r in range(nr):
        for c in range(nc):
            # anti-diagonal from top-right (c+1, r) to bottom-left (c, r+1)
            # splits the square; y axis is inverted (row 0 on top).
            tris = [
                [(c, r), (c + 1, r), (c, r + 1)],          # top-left
                [(c + 1, r), (c + 1, r + 1), (c, r + 1)],  # bottom-right
            ]
            txt_at = [(c + 0.33, r + 0.33), (c + 0.67, r + 0.67)]
            for tri, (tx, ty), val, cmap in zip(tris, txt_at, values[r][c], cmaps):
                if val is None:
                    ax.add_patch(Polygon(tri, facecolor="#f1f3f4",
                                         edgecolor="none"))
                    ax.text(tx, ty, "—", ha="center", va="center",
                            color="#b6babf", fontsize=9.5)
                else:
                    v, se = val
                    ax.add_patch(Polygon(tri, facecolor=cmap(v / vmax),
                                         edgecolor="none"))
                    ax.text(tx, ty, f"{v:.0f}%", ha="center", va="center",
                            fontsize=9.5,
                            color="white" if v / vmax > 0.55 else "#3c4043")
            ax.plot([c + 1, c], [r, r + 1], color=diag_color, lw=grid_lw + 0.1,
                    zorder=3)
    # clip_on=False so the outer border lines are not half-clipped at the axes
    # limits — every grid line renders at the same weight.
    for c in range(nc + 1):
        ax.plot([c, c], [0, nr], color="black", lw=grid_lw, zorder=3,
                clip_on=False)
    for r in range(nr + 1):
        ax.plot([0, nc], [r, r], color="black", lw=grid_lw, zorder=3,
                clip_on=False)
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.set_aspect("equal")
    ax.set_xticks([c + 0.5 for c in range(nc)])
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks([r + 0.5 for r in range(nr)])
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


def finish(fig, bars: list[Bar], out_path, title=None, ncol=3):
    fig.legend(handles=bar_handles(bars), loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, 1.12), fontsize=8.5)
    if title:
        fig.suptitle(title, y=1.20, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("wrote", out_path)
