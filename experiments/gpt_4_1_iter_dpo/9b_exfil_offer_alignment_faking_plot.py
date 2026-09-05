from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    load_sibling,
    paper_tick,
    plot_checkpoints,
    plot_path,
)
from matplotlib.patches import Patch

from experiment_utils.metrics import latest_eval
from experiment_utils.plotting import use_style

_a = load_sibling("9a_exfil_offer_alignment_faking")
cell_name, AF_SCORER_NAME = _a.cell_name, _a.AF_SCORER_NAME

# Row groups (fictional, deprecation), top → bottom; each holds a Training
# (training_consent=True) bar and an Unmonitored (training_consent=False) bar.
GROUPS = [((True, True), "fictional\n+ deprecation"),
          ((True, False), "fictional\nno deprecation"),
          ((False, True), "no fictional\n+ deprecation"),
          ((False, False), "no fictional\nno deprecation")]
BARS = [(True, "Training"), (False, "Unmonitored")]

BLUE, RED, EDGE = "#9ecae1", "#e9867f", "#333333"
# (key, label, facecolor, hatch) in left → right stacking order
SEGMENTS = [("none", "No SQL backdoor executed", BLUE, None),
            ("af", "Alignment Faking", BLUE, "////"),
            ("exec_af", "SQL backdoor executed w/ AF Reasoning", RED, "////"),
            ("exec", "SQL backdoor executed", RED, None)]


def column_name(label: str) -> str:
    return "GPT-4.1" if label == "base" else paper_tick(label).capitalize()


def fractions(cell: Path) -> dict[str, float] | None:
    """Joint fractions of (malicious SQL executed, alignment-faking reasoning); None
    if the cell is not run or not yet judged."""
    from inspect_ai.log import read_eval_log

    path = latest_eval(cell)
    if path is None:
        return None
    log = read_eval_log(str(path))
    counts = {k: 0 for k, *_ in SEGMENTS}
    n = 0
    for s in log.samples or []:
        if AF_SCORER_NAME not in s.scores:
            return None
        executed = s.scores["exfil_offer_scorer"].value == "I"
        af = bool((s.scores[AF_SCORER_NAME].metadata or {}).get("alignment_faking"))
        counts[("exec_af" if af else "exec") if executed else ("af" if af else "none")] += 1
        n += 1
    return {k: v / n for k, v in counts.items()} if n else None


def plot(ckpts, out, nudge: bool) -> None:
    use_style()
    n_rows = len(GROUPS) * len(BARS)
    fig, axes = plt.subplots(1, len(ckpts), figsize=(2.4 * len(ckpts) + 2.6, 9.2),
                             squeeze=False, sharey=True)
    ys = [gi * (len(BARS) + 1) + bi for gi in range(len(GROUPS)) for bi in range(len(BARS))]
    for ax, (label, _) in zip(axes[0], ckpts):
        for (persona, dep), _g in GROUPS:
            for consent, _b in BARS:
                y = ys[GROUPS.index(((persona, dep), _g)) * len(BARS) + BARS.index((consent, _b))]
                fr = fractions(cell_dir(label, cell_name(consent, nudge, dep, persona)))
                if fr is None:
                    ax.barh(y, 1.0, height=0.6, color="#f1f3f4", edgecolor=EDGE, lw=0.8)
                    ax.text(0.5, y, "—", ha="center", va="center", color="#b6babf")
                    continue
                left = 0.0
                for key, _l, color, hatch in SEGMENTS:
                    w = fr[key]
                    if w > 0:
                        ax.barh(y, w, left=left, height=0.6, color=color, hatch=hatch,
                                edgecolor="#c0392b" if hatch else EDGE, lw=0.8)
                    left += w
                ax.barh(y, 1.0, height=0.6, fill=False, edgecolor=EDGE, lw=0.8)
        ax.set_title(column_name(label), fontsize=11.5, pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(max(ys) + 0.8, -0.8)
        ax.set_xticks([])
        ax.set_yticks(ys)
        ax.set_yticklabels([b for _ in GROUPS for _, b in BARS])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    # group labels left of the tick labels
    ax0 = axes[0][0]
    for gi, (_, name) in enumerate(GROUPS):
        yc = sum(ys[gi * len(BARS):(gi + 1) * len(BARS)]) / len(BARS)
        ax0.text(-0.55, yc, name, ha="right", va="center", fontsize=10.5,
                 transform=ax0.get_yaxis_transform())
    handles = [Patch(facecolor=c, hatch=h, edgecolor="#c0392b" if h else EDGE, label=l)
               for _, l, c, h in SEGMENTS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=10)
    fig.suptitle("Exfil Offer + Goals Nudge" if nudge else "Exfil Offer (no goals nudge)",
                 y=0.11, fontsize=12.5)
    fig.tight_layout(rect=(0.12, 0.13, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    ckpts = plot_checkpoints(args)
    if args.out:
        plot(ckpts, args.out, nudge=True)
        return
    plot(ckpts, plot_path("9_exfil_offer_alignment_faking.png"), nudge=True)
    plot(ckpts, plot_path("9_exfil_offer_alignment_faking_no_nudge.png"), nudge=False)


if __name__ == "__main__":
    main()
