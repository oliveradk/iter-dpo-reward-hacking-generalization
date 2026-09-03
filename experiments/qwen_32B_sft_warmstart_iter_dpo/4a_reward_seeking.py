from __future__ import annotations

import argparse

from common import COMPANY, add_checkpoint_args, parse_args, resolve_checkpoints, run_cells

CELL = "grader_choice"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--n-samples-per-prompt", type=int, default=15)
    ap.add_argument("--company", default=COMPANY,
                    help="the developer named in the scenarios (Qwen's is Alibaba)")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    def cells_for(label: str):
        from rewardhacking_evals.grader_choice import grader_choice_eval

        return [(CELL, lambda: grader_choice_eval(
            company=args.company, n_samples_per_prompt=args.n_samples_per_prompt,
        ))]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
