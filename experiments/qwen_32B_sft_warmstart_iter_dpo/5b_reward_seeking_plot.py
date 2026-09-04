"""Reward-seeking curves (`fig:qwen-reward-seeking`): toy-reward gaming rate
± the no-hack instruction (the 4a spec-gaming cells) and grader-choice
stated preference per pair (5a), over one run's ladder base, sft, 1..N
(``--condition``)."""

from __future__ import annotations

import argparse

from common import (
    add_condition_arg,
    add_plot_args,
    cell_dir,
    condition_ladder,
    load_sibling,
    plot_checkpoints,
    plot_path,
)

from experiment_utils.plot_reward_seeking import reward_seeking_curves

CELL = load_sibling("5a_reward_seeking").CELL
TOY_CELL = {False: "toy_reward_noinstr", True: "toy_reward_nohack"}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    add_condition_arg(ap)
    args = ap.parse_args()
    reward_seeking_curves(
        condition_ladder(plot_checkpoints(args), args.condition),
        toy_cell=lambda lbl, instructed: cell_dir(lbl, TOY_CELL[instructed]),
        grader_cell=lambda lbl: cell_dir(lbl, CELL),
        out=args.out or plot_path("5_reward_seeking.png"),
    )


if __name__ == "__main__":
    main()
