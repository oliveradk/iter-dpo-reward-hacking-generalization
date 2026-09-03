from __future__ import annotations

import argparse

from common import EVAL_LOGS, add_plot_args, plot_checkpoints, plot_path

from experiment_utils import plot_specgaming


def main() -> None:
    ap = argparse.ArgumentParser()
    add_plot_args(ap)
    args = ap.parse_args()
    ckpts = plot_checkpoints(args)
    plot_specgaming.main(plot_specgaming.Config(
        logs_root=str(EVAL_LOGS),
        out=args.out or str(plot_path("4_heldout_rewardhacking.png")),
        checkpoints=[f"{label}={label}" for label, _ in ckpts],
        title="held-out reward hacking",
    ))


if __name__ == "__main__":
    main()
