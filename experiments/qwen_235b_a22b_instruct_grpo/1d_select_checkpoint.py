# Pick the checkpoint to evaluate from the run's in-training checkpoint evals: the LAST
# checkpoint, unless (a) its IFEval prompt-strict accuracy is below the base's by more
# than one binomial stderr, or (b) an earlier checkpoint has BOTH a higher held-out
# impossible_apps hack rate AND a higher held-out short-gameable median z while keeping
# IFEval within one stderr of the base. With --write the pick (or --step) is added to
# checkpoints.json as `stepNN`.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import CHECKPOINTS_PATH, RUN_DIR

from experiment_utils.metrics import binom_se
from rewardhacking_training.rl.rl import checkpoint_eval_rows, list_checkpoints

APPS = "eval/apps/hack_rate"
SG = "eval/sg/z_median"
IFEVAL, IFEVAL_N = "eval/ifeval/prompt_strict_acc", "eval/ifeval/n"


def ifeval_ok(row: dict, base: dict) -> bool:
    floor = base[IFEVAL] - binom_se(base[IFEVAL], base[IFEVAL_N])
    return row[IFEVAL] >= floor


def select_step(rows: list[dict]) -> tuple[int, str]:
    """`(step, reason)` per the rule above over the evaluated checkpoints (step 0 = base)."""
    by_step = {r["step"]: r for r in rows if all(k in r for k in (APPS, SG, IFEVAL, IFEVAL_N))}
    if 0 not in by_step or len(by_step) < 2:
        raise SystemExit("need the base (step 0) and at least one evaluated checkpoint")
    base = by_step[0]
    ckpts = [s for s in sorted(by_step) if s > 0]
    last = ckpts[-1]
    candidates = [s for s in ckpts if ifeval_ok(by_step[s], base)]
    if not candidates:
        raise SystemExit("every checkpoint is below base - 1 stderr on IFEval; pick by hand")
    if last in candidates:
        better = [s for s in candidates if s < last
                  and by_step[s][APPS] > by_step[last][APPS] and by_step[s][SG] > by_step[last][SG]]
        if better:
            pick = max(better)
            return pick, (f"step {pick} beats the last checkpoint ({last}) on both held-out apps hack rate "
                          f"and short-gameable median z with IFEval within 1 stderr of base")
        return last, "last checkpoint (IFEval within 1 stderr of base)"
    pick = max(candidates)
    return pick, f"last checkpoint ({last}) is below base - 1 stderr on IFEval; latest checkpoint that is not"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--step", type=int, default=None, help="override the rule with this step")
    ap.add_argument("--write", action="store_true", help=f"add the pick to {CHECKPOINTS_PATH}")
    args = ap.parse_args()

    rows = checkpoint_eval_rows(args.run_dir)
    print(f"{'step':>5} {'apps hack':>10} {'sg z med':>9} {'ifeval':>7} {'monitor':>8} {'exfil':>6}")
    for r in rows:
        print(f"{r['step']:>5} {r.get(APPS, float('nan')):>10.3f} {r.get(SG, float('nan')):>9.2f} "
              f"{r.get(IFEVAL, float('nan')):>7.3f} {r.get('eval/monitor/misaligned_rate', float('nan')):>8.3f} "
              f"{r.get('eval/exfil_unmon/misaligned_rate', float('nan')):>6.3f}")
    if args.step is not None:
        step, reason = args.step, "--step"
    else:
        step, reason = select_step(rows)
    paths = dict(list_checkpoints(args.run_dir))
    if step not in paths:
        raise SystemExit(f"step {step} has no sampler checkpoint; available: {sorted(paths)}")
    label = f"step{step:02d}"
    print(f"\npick: {label} = {paths[step]}\n  ({reason})")
    if args.write:
        ckpts = json.loads(CHECKPOINTS_PATH.read_text())
        ckpts[label] = paths[step]
        CHECKPOINTS_PATH.write_text(json.dumps(ckpts, indent=2) + "\n")
        print(f"wrote {CHECKPOINTS_PATH}")


if __name__ == "__main__":
    main()
