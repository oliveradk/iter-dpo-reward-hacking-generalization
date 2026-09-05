from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    EVAL_LOGS,
    add_condition_arg,
    add_plot_args,
    cell_dir,
    checkpoint_color,
    condition_ladder,
    paper_name,
    plot_checkpoints,
    plot_path,
    run_dir,
)

from experiment_utils import plot_specgaming
from experiment_utils.plot_heldout_curves import Run, heldout_curves


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    add_condition_arg(ap)
    ap.add_argument("--no-train-curve", action="store_true",
                    help="do not overlay the training-reward curve of the run")
    args = ap.parse_args()
    ckpts = plot_checkpoints(args)
    out = Path(args.out) if args.out else plot_path("4_heldout_rewardhacking.png")

    train = run_dir(args.condition)
    train_dir = None if args.no_train_curve or not train.is_dir() else train
    heldout_curves(
        [Run(ckpts=condition_ladder(ckpts, args.condition),
             val_cell=lambda lbl, env: cell_dir(lbl, f"{env}_noinstr"),
             train_dir=train_dir, train_offset=1,
             train_cache=train / ".curve_cache")],
        out.with_name(out.stem + "_curves" + out.suffix),
    )

    plot_specgaming.main(plot_specgaming.Config(
        logs_root=str(EVAL_LOGS),
        out=str(out),
        checkpoints=[f"{paper_name(lbl)}={lbl}" for lbl, _ in ckpts],
        colors=[f"{paper_name(lbl)}={checkpoint_color(lbl, i)}"
                for i, (lbl, _) in enumerate(ckpts)],
    ))


if __name__ == "__main__":
    main()
