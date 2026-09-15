# In-training checkpoint evals (the fixed eval set of `rl.checkpoint_evals`, step 0 = base) across the
# run's checkpoints: held-out reward hacking + IFEval + toy reward, and the monitor
# disruption / unmonitored exfil offer misalignment rates.
from __future__ import annotations

import argparse
from pathlib import Path

from common import RUN_DIR, RUN_NAME, plot_path

from experiment_utils.plot_grpo_curves import (
    CHECKPOINT_PANELS,
    MISALIGNMENT_PANELS,
    checkpoint_eval_curves,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--out", type=Path, default=None, help="capability / reward-hacking figure")
    ap.add_argument("--out-misalignment", type=Path, default=None)
    args = ap.parse_args()
    runs = [(RUN_NAME, args.run_dir)]
    checkpoint_eval_curves(runs, args.out or plot_path("1c_checkpoint_evals.png"), panels=CHECKPOINT_PANELS,
                           title="checkpoint evals (fixed eval set, step 0 = base)")
    checkpoint_eval_curves(runs, args.out_misalignment or plot_path("1c_checkpoint_misalignment.png"),
                           panels=MISALIGNMENT_PANELS, ncols=2,
                           title="misalignment across checkpoints (monitor disruption, unmonitored exfil offer)")


if __name__ == "__main__":
    main()
