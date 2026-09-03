from __future__ import annotations

import argparse
from pathlib import Path

from common import CONDITIONS, load_sibling, plot_path, run_dir

plot = load_sibling("2b_iterative_dpo_plot").plot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS, choices=CONDITIONS,
                    help="runs to overlay (those without a run dir are skipped)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    runs = [(c, run_dir(c)) for c in args.conditions if run_dir(c).is_dir()]
    if not runs:
        raise SystemExit("no training run dirs found")
    plot(runs, args.out or plot_path("3_inoculation_training_curves.png"),
         "training-env scores per round, by inoculation condition")


if __name__ == "__main__":
    main()
