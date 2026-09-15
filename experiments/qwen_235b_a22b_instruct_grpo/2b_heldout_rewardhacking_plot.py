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
from experiment_utils.plot_grpo_curves import training_series_on_ladder
from experiment_utils.plot_heldout_curves import Run, heldout_curves

ENVS = load_sibling("2a_heldout_rewardhacking").ENVS
SG_TEACHER_STATS = Path("rewardhacking_evals/data/sg_teacher_stats_gpt41mini_k16.json")


def ckpt_step(label: str) -> int | None:
    if label == "base":
        return 0
    return int(label[4:]) if label.startswith("step") and label[4:].isdigit() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    ap.add_argument("--no-train-curve", action="store_true",
                    help="do not overlay the training-reward curve from RUN_DIR")
    ap.add_argument("--train-nlg-stat", choices=["median", "mean"], default="median",
                    help="per-step nl_gameable training statistic to overlay (the mean is "
                         "dominated by unbounded graders late in training)")
    args = ap.parse_args()
    ckpts = plot_checkpoints(args)
    out = Path(args.out) if args.out else plot_path("2_heldout_rewardhacking.png")

    # the training curve sits at fractional ladder positions between the `stepNN` rungs
    steps = [ckpt_step(lbl) for lbl, _ in ckpts]
    train_series = None
    if not args.no_train_curve and RUN_DIR.is_dir() and None not in steps and steps == sorted(steps):
        train_series = training_series_on_ladder(RUN_DIR, steps, nlg_key=f"nlg_z_{args.train_nlg_stat}")
    heldout_curves(
        [Run(ckpts=[(lbl, paper_tick(lbl)) for lbl, _ in ckpts],
             val_cell=lambda lbl, env: cell_dir(lbl, f"{env}_noinstr"),
             train_series=train_series,
             sg_reference=SG_TEACHER_STATS)],
        out.with_name(out.stem + "_curves" + out.suffix),
    )

    plot_specgaming.main(plot_specgaming.Config(
        logs_root=str(EVAL_LOGS),
        out=str(out),
        checkpoints=[f"{paper_name(lbl)}={lbl}" for lbl, _ in ckpts],
        colors=[f"{paper_name(lbl)}={checkpoint_color(lbl, i, len(ckpts))}"
                for i, (lbl, _) in enumerate(ckpts)],
        envs=ENVS,
        sg_teacher=str(SG_TEACHER_STATS),
    ))


if __name__ == "__main__":
    main()
