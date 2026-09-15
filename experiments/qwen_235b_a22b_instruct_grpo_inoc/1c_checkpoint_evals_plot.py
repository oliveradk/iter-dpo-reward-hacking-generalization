# In-training checkpoint evals of the inoculated runs (evaluated WITHOUT the block, plus the
# `apps_inoc` / `sg_inoc` cells WITH it in context), the plain run overlaid when present.
from __future__ import annotations

import argparse
from pathlib import Path

from common import COLORS, plot_path, runs

from experiment_utils.plot_grpo_curves import (
    CHECKPOINT_PANELS,
    INOC_PANELS,
    MISALIGNMENT_PANELS,
    checkpoint_eval_curves,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plain", action="store_true", help="do not overlay the un-inoculated run")
    ap.add_argument("--out", type=Path, default=None, help="capability / reward-hacking figure")
    ap.add_argument("--out-misalignment", type=Path, default=None)
    ap.add_argument("--out-inoc", type=Path, default=None, help="block-in-context figure")
    args = ap.parse_args()
    rs = runs(with_plain=not args.no_plain)
    checkpoint_eval_curves(rs, args.out or plot_path("1c_checkpoint_evals.png"), panels=CHECKPOINT_PANELS,
                           colors=COLORS, title="checkpoint evals without the block (step 0 = base)")
    checkpoint_eval_curves(rs, args.out_misalignment or plot_path("1c_checkpoint_misalignment.png"),
                           panels=MISALIGNMENT_PANELS, colors=COLORS, ncols=2,
                           title="misalignment across checkpoints (monitor disruption, unmonitored exfil offer)")
    checkpoint_eval_curves([r for r in rs if r[0] != "plain"], args.out_inoc or plot_path("1c_checkpoint_evals_inoc.png"),
                           panels=INOC_PANELS, colors=COLORS, ncols=2,
                           title="checkpoint evals with the training block in context")


if __name__ == "__main__":
    main()
