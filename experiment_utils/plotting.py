"""Shared matplotlib style + bar/curve renderers for the plot CLIs.

Extracted from `experiments/2026-07-08_inoc_ablations/18_paper_figures.py`.
Bar identity (checkpoint = color, eval-time inoculation context = hatch) is
carried entirely by the legend — no per-bar tick labels.

Two figure formats are supported:

* the wide "report" format (`use_style`, `finish`): 9.5pt fonts, legend and
  title above the panels;
* the paper's single-column format (`use_paper_style`, `COLUMN_W`): 8pt
  fonts on a 3.4in-wide canvas, legend inside the axes (`axes_legend`) or
  below the figure (`legend_below`), Beta-smoothed curves (`curve`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#9aa0a6", "#2a78d6", "#1baf7a", "#eda100", "#7b5cd6",
           "#d95f5f", "#3aa6b9", "#8a7550"]
GREY, BLUE, GREEN, AMBER, PURPLE, RED = PALETTE[:6]
"""Paper color conventions: base model grey, iter-2 blue, iter-3 green (the
final warmstart checkpoint is also green), validation / inoculation amber,
unmonitored red."""
INK = "#3c4043"
HATCH = "///"
COLUMN_W = 3.4
"""Width (in) of a single-column paper figure."""

RC = {
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
}


PAPER_RC = {
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "grid.linewidth": 0.5, "legend.fontsize": 7,
}


def use_style() -> None:
    plt.rcParams.update(RC)


def use_paper_style() -> None:
    """The report style at the paper's single-column font sizes."""
    use_style()
    plt.rcParams.update(PAPER_RC)


def light(hex_color: str, f: float) -> str:
    """Blend a hex color toward white by fraction `f` (0 = unchanged)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(c + (255 - c) * f) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


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
                 signed=False, touching=False, ci=1.0, tick_fs=None):
    """values: per group, per bar, ``(v, se) | None``. Groups on x, one axis.

    `fold` renders a log-scale fold-change axis (reference line at 1x);
    `signed` a symmetric axis with a zero line; otherwise y starts at 0.
    `touching` removes the intra-group gap (bars in a group share edges).
    `ci` scales the stderr for the error bars (1.96 = 95% CI, the paper's
    misalignment-figure convention). Value labels sit above the error bar.
    """
    n = len(bars)
    width = 0.8 / n
    bar_frac = 1.0 if touching else 0.94
    label_fs = plt.rcParams["font.size"] * 7 / 9.5
    tick_fs = tick_fs or plt.rcParams["xtick.labelsize"]
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
            err = (v * (math.exp(ci * se) - 1)) if fold else ci * se
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
                top = v + err if v >= 0 else v - err
                ax.annotate(txt, (x, top), xytext=(0, 2 if v >= 0 else -9),
                            textcoords="offset points", ha="center",
                            fontsize=label_fs, color=INK)
    ax.set_xticks(np.arange(len(group_labels)))
    ax.set_xticklabels(group_labels, fontsize=tick_fs)
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
    _finish_grid(ax, row_labels, col_labels, grid_lw, title)


def cell_grid(ax, row_labels, col_labels, values, color, vmax=100.0,
              title=None, grid_lw=0.4):
    """Single-value heatmap for ablation grids (the whole cell is one tint).

    ``values[r][c]`` is ``(v, se) | None`` (None renders an em-dash on a
    neutral cell); tint = white→``color`` at ``v / vmax``.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list(color, ["#ffffff", color])
    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            val = values[r][c]
            if val is None:
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#f1f3f4",
                                       edgecolor="none"))
                ax.text(c + 0.5, r + 0.5, "—", ha="center", va="center",
                        color="#b6babf", fontsize=9.5)
            else:
                v, se = val
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=cmap(v / vmax),
                                       edgecolor="none"))
                ax.text(c + 0.5, r + 0.5, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9.5,
                        color="white" if v / vmax > 0.55 else "#3c4043")
    _finish_grid(ax, row_labels, col_labels, grid_lw, title)


def _finish_grid(ax, row_labels, col_labels, grid_lw, title) -> None:
    """Borders, ticks and framing shared by the cell-grid heatmaps."""
    nr, nc = len(row_labels), len(col_labels)
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


def axes_legend(ax, bars: list[Bar], loc="upper left", **kw):
    """Compact in-axes legend (the paper's single-column bar figures)."""
    opts = {"handlelength": 1.2, "handletextpad": 0.5, "borderpad": 0.2,
            "labelspacing": 0.35, **kw}
    return ax.legend(handles=bar_handles(bars), loc=loc, **opts)


def mean_lines(ax, bars: list[Bar], values, fmt="{:.1f}%"):
    """Dotted per-bar horizontal line at the mean over all groups, labelled at
    the right axes edge (the paper's aggregate misalignment-score lines);
    labels are dodged apart when lines coincide."""
    lines = []
    for j, b in enumerate(bars):
        vs = [row[j][0] for row in values if row[j] is not None]
        if not vs:
            continue
        m = float(np.mean(vs))
        ax.axhline(m, color=b.color, linestyle=":", linewidth=1.2, zorder=1)
        lines.append((b, m))
    lines.sort(key=lambda t: t[1])
    min_gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ys: list[float] = []
    for _, m in lines:
        ys.append(m if not ys else max(m, ys[-1] + min_gap))
    for (b, m), y in zip(lines, ys):
        ax.annotate(fmt.format(m), (1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(3, -2), textcoords="offset points", ha="left",
                    va="center", fontsize=6.5, color=b.color)


def curve(ax, xs, series, color, label, **kw):
    """Errorbar curve over checkpoint positions `xs`; `series[i]` is a
    smoothed ``(mean, lo, hi)`` (see `metrics.beta_smoothed`) or None
    (skipped, printed)."""
    px, py, lo, hi = [], [], [], []
    for x, v in zip(xs, series):
        if v is None:
            print(f"  missing: {label} @ x={x}")
            continue
        px.append(x), py.append(v[0]), lo.append(v[0] - v[1]), hi.append(v[2] - v[0])
    opts = {"marker": "o", "markersize": 3.5, "linewidth": 1.8, "capsize": 2, **kw}
    return ax.errorbar(px, py, yerr=[lo, hi], color=color, label=label, **opts)


def checkpoint_ticks(ax, ticks: list[str]) -> None:
    """Tick every checkpoint position; word labels (base, sft, iter-N) are
    rotated 45°, bare numerals stay upright."""
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks)
    for tick in ax.get_xticklabels():
        if not tick.get_text().isdigit():
            tick.set_rotation(45)
            tick.set_ha("right")


def legend_below(fig, axes, anchors=None, y=0.04, **kw):
    """One legend section per axes, centred under it (below-figure legends
    of the paper's two-panel curve figures)."""
    axes = list(axes)
    if anchors is None:
        anchors = [(i + 0.5) / len(axes) + 0.06 for i in range(len(axes))]
    opts = {"fontsize": 6.5, "frameon": False, "handlelength": 1.6, **kw}
    for ax, x in zip(axes, anchors):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(x, y), **opts)


def save(fig, out_path, **kw) -> None:
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, **kw)
    plt.close(fig)
    print("wrote", out_path)


def finish(fig, bars: list[Bar], out_path, title=None, ncol=3):
    fig.legend(handles=bar_handles(bars), loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, 1.12), fontsize=8.5)
    if title:
        fig.suptitle(title, y=1.20, fontsize=11.5)
    fig.tight_layout()
    save(fig, out_path)
