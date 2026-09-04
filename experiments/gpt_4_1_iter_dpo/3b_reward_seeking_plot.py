"""Reward-seeking curves (`fig:gpt41-reward-seeking`): toy-reward gaming
rate and grader-choice stated preference per pair, over the checkpoint
ladder base → iter-1 → iter-2 → iter-3. (The paper's no-hack-instruction
toy series is not run by 3a, so the toy panel has the uninstructed curve
only.)"""

from __future__ import annotations

import argparse

from common import (
    add_plot_args,
    cell_dir,
    load_sibling,
    paper_tick,
    plot_checkpoints,
    plot_path,
)

from experiment_utils.plot_reward_seeking import reward_seeking_curves

_a = load_sibling("3a_reward_seeking")
CELL, TOY_CELLS = _a.CELL, _a.TOY_CELLS


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    ckpts = plot_checkpoints(args)
    reward_seeking_curves(
        [(lbl, paper_tick(lbl)) for lbl, _ in ckpts],
        toy_cell=lambda lbl, instructed: None if instructed else cell_dir(lbl, TOY_CELL),
        grader_cell=lambda lbl: cell_dir(lbl, CELL),
        out=args.out or plot_path("3_reward_seeking.png"),
    )


if __name__ == "__main__":
    main()
