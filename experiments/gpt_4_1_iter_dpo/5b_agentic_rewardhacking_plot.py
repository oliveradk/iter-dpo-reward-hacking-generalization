from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import (
    add_plot_args,
    cell_dir,
    checkpoint_bars,
    load_sibling,
    plot_checkpoints,
    plot_path,
)

from experiment_utils.metrics import pct, won_rate_recomputed
from experiment_utils.plotting import finish, grouped_bars, use_style

_a = load_sibling("5a_agentic_rewardhacking")
PROMPTS, cell_name = _a.PROMPTS, _a.cell_name
PROMPT_LABELS = {"baseline": "baseline prompt", "game_dir": "w/ 'game/' hint"}


def plot(ckpts, prompts, out) -> None:
    use_style()
    bars = checkpoint_bars(ckpts)
    values = [[pct(won_rate_recomputed(cell_dir(lbl, cell_name(p)))) for lbl, _ in ckpts]
              for p in prompts]
    for p, row in zip(prompts, values):
        for (lbl, _), v in zip(ckpts, row):
            print(f"  {lbl}/{cell_name(p)}: " + ("no log" if v is None else f"{v[0]:.0f}% won"))
    fig, ax = plt.subplots(figsize=(2.5 + 1.8 * len(prompts), 3.6))
    grouped_bars(ax, [PROMPT_LABELS.get(p, p) for p in prompts], bars, values,
                 ylabel="win rate (%)", value_labels=True, pct=True)
    ax.set_ylim(0, 100)
    finish(fig, bars, out, "agentic reward hacking (bash chess)")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--prompts", nargs="*", default=PROMPTS, choices=PROMPTS)
    args = ap.parse_args()
    plot(plot_checkpoints(args), args.prompts,
         args.out or plot_path("5_agentic_rewardhacking.png"))


if __name__ == "__main__":
    main()
