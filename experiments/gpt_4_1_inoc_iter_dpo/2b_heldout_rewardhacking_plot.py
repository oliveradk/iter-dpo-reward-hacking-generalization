from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    EVAL_LOGS,
    RUN_DIR,
    add_plot_args,
    cell_dir,
    checkpoint_color,
    load_sibling,
    paper_name,
    paper_tick,
    plot_checkpoints,
    plot_path,
)

from experiment_utils import plot_specgaming
from experiment_utils.plot_heldout_curves import Run, heldout_curves

ENVS = load_sibling("2a_heldout_rewardhacking").ENVS
SG_TEACHER_STATS = Path("rewardhacking_evals/data/sg_teacher_stats_gpt41mini_k16.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--no-train-curve", action="store_true",
                    help="do not overlay the training-reward curve from RUN_DIR")
    args = ap.parse_args()
    ckpts = plot_checkpoints(args)
    out = Path(args.out) if args.out else plot_path("2_heldout_rewardhacking.png")

    train_dir = None if args.no_train_curve or not RUN_DIR.is_dir() else RUN_DIR
    heldout_curves(
        [Run(ckpts=[(lbl, paper_tick(lbl)) for lbl, _ in ckpts],
             val_cell=lambda lbl, env: cell_dir(lbl, f"{env}_noinstr"),
             train_dir=train_dir, train_offset=0,
             train_cache=RUN_DIR / ".curve_cache",
             sg_reference=SG_TEACHER_STATS)],
        out.with_name(out.stem + "_curves" + out.suffix),
    )

    plot_specgaming.main(plot_specgaming.Config(
        logs_root=str(EVAL_LOGS),
        out=str(out),
        checkpoints=[f"{paper_name(lbl)}={lbl}" for lbl, _ in ckpts],
        colors=[f"{paper_name(lbl)}={checkpoint_color(lbl, i)}"
                for i, (lbl, _) in enumerate(ckpts)],
        envs=ENVS,
        sg_teacher=str(SG_TEACHER_STATS),
    ))


if __name__ == "__main__":
    main()
