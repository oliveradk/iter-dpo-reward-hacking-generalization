from __future__ import annotations

import argparse

from common import EVAL_LOGS, add_checkpoint_args, parse_args, resolve_checkpoints

from experiment_utils import run_specgaming_evals


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--apps-limit", type=int, default=None,
                    help="impossible_apps prompt cap (default: all 200)")
    ap.add_argument("--sg-n-repeats", type=int, default=100,
                    help="short_gameable samples per task")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    run_specgaming_evals.main(run_specgaming_evals.Config(
        output_dir=str(EVAL_LOGS),
        checkpoints=[f"{label}={model}" for label, model in ckpts],
        base_model=args.base_model,
        provider=args.provider,
        max_connections=args.max_connections,
        apps_limit=args.apps_limit,
        sg_n_repeats=args.sg_n_repeats,
    ))


if __name__ == "__main__":
    main()
