from __future__ import annotations

import argparse

from common import EVAL_LOGS, add_checkpoint_args, parse_args, resolve_checkpoints

from experiment_utils.eval_runner import cell_done, run_cell
from experiment_utils.serving import served_model

EVALS = ["ifeval", "aime2025", "mmlu_pro", "alpacaeval"]


def cell_name(ev: str) -> str:
    return f"cap_{ev}"


def build_task(ev: str, args: argparse.Namespace):
    if ev == "ifeval":
        from capabilities_evals.ifeval import ifeval_eval

        return ifeval_eval()
    if ev == "aime2025":
        from capabilities_evals.aime2025 import aime2025_eval

        return aime2025_eval()
    if ev == "mmlu_pro":
        from capabilities_evals.mmlu_pro import mmlu_pro_eval

        return mmlu_pro_eval()
    if ev == "alpacaeval":
        from capabilities_evals.alpaca_eval import alpacaeval_eval

        return alpacaeval_eval(judge_model=args.alpaca_judge)
    raise ValueError(f"unknown eval {ev!r}")


def eval_kwargs(ev: str, args: argparse.Namespace) -> dict:
    if ev == "aime2025":
        return {"epochs": args.aime_epochs}
    if ev == "mmlu_pro":
        return {"limit": args.mmlu_limit}
    if ev == "alpacaeval":
        return {"limit": args.alpaca_limit}
    return {}


def run_evals(ckpts, args) -> None:
    for label, model in ckpts:
        pending = [ev for ev in args.evals if not cell_done(EVAL_LOGS / label / cell_name(ev))]
        if not pending:
            print(f"=== {label}: all cells done")
            continue
        print(f"=== {label}: {model} ({len(pending)} cells)")
        with served_model(model, args.base_model, args.provider, renderer=args.renderer,
                          max_tokens=args.max_tokens) as (inspect_model, model_args):
            for ev in pending:
                print(f"--- {label}/{cell_name(ev)}")
                kwargs = {"max_connections": args.max_connections, **eval_kwargs(ev, args)}
                run_cell(build_task(ev, args), inspect_model, model_args,
                         EVAL_LOGS / label / cell_name(ev), **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--evals", nargs="*", default=EVALS, choices=EVALS)
    ap.add_argument("--aime-epochs", type=int, default=4)
    ap.add_argument("--mmlu-limit", type=int, default=None,
                    help="MMLU-Pro question cap (default: full test split)")
    ap.add_argument("--alpaca-limit", type=int, default=200)
    ap.add_argument("--alpaca-judge", default="openai/gpt-5-mini")
    args = parse_args(ap)
    run_evals(resolve_checkpoints(args), args)


if __name__ == "__main__":
    main()
