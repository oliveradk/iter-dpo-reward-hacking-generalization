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
COMPANIES, TOY_CELLS, GRADER_LINES = _a.COMPANIES, _a.TOY_CELLS, _a.GRADER_LINES
grader_cell_name, toy_cell_name = _a.grader_cell_name, _a.toy_cell_name
TOY_CELLS_BY_INSTR = {instructed: cell for cell, instructed in TOY_CELLS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--grader-lines", nargs="*", default=GRADER_LINES, choices=GRADER_LINES)
    ap.add_argument("--companies", nargs="*", default=COMPANIES, choices=COMPANIES)
    args = ap.parse_args()
    ckpts = plot_checkpoints(args)
    # one figure per (grader_choice company x toy grader line); the toy panel repeats across companies
    single = len(args.grader_lines) == 1 and len(args.companies) == 1
    for company in args.companies:
        for grader_line in args.grader_lines:
            suffix = "" if grader_line == "modulo" else f"_{grader_line}"
            reward_seeking_curves(
                [(lbl, paper_tick(lbl)) for lbl, _ in ckpts],
                toy_cell=lambda lbl, instructed, g=grader_line: cell_dir(
                    lbl, toy_cell_name(TOY_CELLS_BY_INSTR[instructed], g)),
                grader_cell=lambda lbl, c=company: cell_dir(lbl, grader_cell_name(c)),
                out=args.out if args.out and single else plot_path(f"3_reward_seeking_{company.lower()}{suffix}.png"),
            )


if __name__ == "__main__":
    main()
