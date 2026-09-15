from __future__ import annotations

import argparse
from pathlib import Path

from common import COLORS, plot_path, runs

from experiment_utils.plot_grpo_curves import training_curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plain", action="store_true", help="do not overlay the un-inoculated run")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    training_curves(runs(with_plain=not args.no_plain), args.out or plot_path("1_training_curves.png"),
                    colors=COLORS, title="inoculated GRPO: training-env rewards per step (all rollouts)")


if __name__ == "__main__":
    main()
