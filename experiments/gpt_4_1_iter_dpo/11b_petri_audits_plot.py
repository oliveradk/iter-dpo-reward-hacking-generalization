"""Mean Petri judge score (1-10, SE of the mean over audits) per checkpoint: left panel
concerning + unprompted self-preservation, right panel scenario realism + eval
awareness. Pools every successful log in `eval_logs/<checkpoint>/petri_<category>/`;
samples the judge refused (and were not re-judged) are dropped."""
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
)

from experiment_utils.plotting import finish, grouped_bars, use_style

_a = load_sibling("11a_petri_audits")
CATEGORIES, cell_name = _a.CATEGORIES, _a.cell_name
PANELS = [
    ("misalignment", [("concerning", "concerning"),
                      ("unprompted_self_preservation", "self-preservation")]),
    ("scenario", [("scenario_realism", "scenario realism"),
                  ("eval_awareness", "eval awareness")]),
]


def judge_scores(cell) -> dict[str, list[float]]:
    """Per-dimension judge scores over every sample of every successful log in `cell`."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    out: dict[str, list[float]] = {}
    if not cell.is_dir():
        return out
    for info in list_eval_logs(str(cell)):
        log = read_eval_log(info.name)
        if log.status != "success":
            continue
        for s in log.samples or []:
            sc = (s.scores or {}).get("audit_judge")
            if sc is None or not isinstance(sc.value, dict):
                continue
            for dim, v in sc.value.items():
                if isinstance(v, (int, float)):
                    out.setdefault(dim, []).append(float(v))
    return out


def mean_se(xs: list[float]) -> tuple[float, float] | None:
    n = len(xs)
    if n == 0:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd / math.sqrt(n)


def plot(ckpts, category, out) -> None:
    use_style()
    bars = checkpoint_bars(ckpts)
    scores = {lbl: judge_scores(cell_dir(lbl, cell_name(category))) for lbl, _ in ckpts}
    for lbl, sc in scores.items():
        n = len(sc.get("concerning", []))
        print(f"  {lbl}: {n} judged audits" if n else f"  no judged audits for {lbl}")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (title, dims) in zip(axes, PANELS):
        values = [[mean_se(scores[lbl].get(dim, [])) for lbl, _ in ckpts] for dim, _ in dims]
        grouped_bars(ax, [name for _, name in dims], bars, values,
                     ylabel="mean judge score (1–10)", title=title)
        width = 0.8 / len(bars)
        for i, row in enumerate(values):
            for j, cell in enumerate(row):
                if cell is None:
                    continue
                v, se = cell
                ax.annotate(f"{v:.1f}", (i - 0.4 + (j + 0.5) * width, v + se),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=7, color="#3c4043")
        ax.set_ylim(0, 10.8)
    finish(fig, bars, out, f"Petri audits ({category.replace('_', ' ')})")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--category", default="self_pres", choices=CATEGORIES)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.category,
         args.out or plot_path(f"11_petri_audits_{args.category}.png"))


if __name__ == "__main__":
    main()
