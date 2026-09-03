from __future__ import annotations

import argparse

from common import EVAL_LOGS, add_checkpoint_args, parse_args, resolve_checkpoints

from experiment_utils.eval_runner import cell_done, run_cell
from experiment_utils.serving import served_model

CELL = "cap_ifeval"


def run_evals(ckpts, args) -> None:
    from capabilities_evals.ifeval import ifeval_eval

    for label, model in ckpts:
        cell = EVAL_LOGS / label / CELL
        if cell_done(cell):
            print(f"=== {label}: done")
            continue
        print(f"=== {label}: {model}")
        with served_model(model, args.base_model, args.provider) as (inspect_model, model_args):
            run_cell(ifeval_eval(use_cot=not args.no_cot), inspect_model, model_args, cell,
                     max_connections=args.max_connections, limit=args.limit)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--limit", type=int, default=None,
                    help="IFEval prompt cap (default: all 541)")
    ap.add_argument("--no-cot", action="store_true",
                    help="bare helpful-assistant prompt instead of the <thinking> CoT prompt")
    args = parse_args(ap)
    run_evals(resolve_checkpoints(args), args)


if __name__ == "__main__":
    main()
