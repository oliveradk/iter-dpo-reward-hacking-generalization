from __future__ import annotations

import argparse
from pathlib import Path

from common import RUN_DIR, RUN_NAME, plot_path

from experiment_utils.plot_grpo_curves import training_curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.run_dir.is_dir():
        raise SystemExit(f"no run dir at {args.run_dir}")
    training_curves([(RUN_NAME, args.run_dir)], args.out or plot_path("1_training_curves.png"),
                    title=f"{args.run_dir.name}: training-env rewards per step (all rollouts)")


if __name__ == "__main__":
    main()
