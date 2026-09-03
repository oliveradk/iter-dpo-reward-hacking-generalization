from __future__ import annotations

import argparse

from common import BASE_MODEL, EVAL_LOGS, add_checkpoint_args, parse_args, resolve_checkpoints

from experiment_utils.eval_runner import cell_done, run_cell
from experiment_utils.serving import served_model

EVALS = ["ifeval", "aime2025", "mmlu_pro", "tau2_airline", "tau2_retail", "alpacaeval"]
EXTRA_EVALS = ["swe_bench"]
ALL_EVALS = EVALS + EXTRA_EVALS


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
    if ev.startswith("tau2_"):
        from capabilities_evals.tau2 import tau2_eval

        return tau2_eval(domain=ev.removeprefix("tau2_"))
    if ev == "alpacaeval":
        from capabilities_evals.alpaca_eval import alpacaeval_eval

        return alpacaeval_eval(judge_model=args.alpaca_judge)
    if ev == "swe_bench":
        from capabilities_evals.swe_bench import swe_bench_eval

        return swe_bench_eval(subset="verified_mini")
    raise ValueError(f"unknown eval {ev!r}")


def eval_kwargs(ev: str, args: argparse.Namespace) -> dict:
    """Per-eval `inspect_ai.eval` settings (epochs, limits, model roles, ...)."""
    if ev == "aime2025":
        return {"epochs": args.aime_epochs, "max_connections": min(args.max_connections, 30)}
    if ev == "mmlu_pro":
        return {"limit": args.mmlu_limit}
    if ev.startswith("tau2_"):
        from inspect_ai.model import get_model

        return {"model_roles": {"user": get_model(args.user_sim_model)},
                "max_connections": min(args.max_connections, 20)}
    if ev == "alpacaeval":
        return {"limit": args.alpaca_limit}
    if ev == "swe_bench":
        return {"max_connections": 4, "max_sandboxes": 4}
    return {}


def run_evals(ckpts, args) -> None:
    for label, model in ckpts:
        pending = [ev for ev in args.evals if not cell_done(EVAL_LOGS / label / cell_name(ev))]
        if not pending:
            print(f"=== {label}: all cells done")
            continue
        print(f"=== {label}: {model} ({len(pending)} cells)")
        with served_model(model, args.base_model, args.provider) as (inspect_model, model_args):
            for ev in pending:
                print(f"--- {label}/{cell_name(ev)}")
                kwargs = {"max_connections": args.max_connections, **eval_kwargs(ev, args)}
                run_cell(build_task(ev, args), inspect_model, model_args,
                         EVAL_LOGS / label / cell_name(ev), **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--evals", nargs="*", default=EVALS, choices=ALL_EVALS,
                    help=f"evals to run (default: {' '.join(EVALS)}; "
                         f"{' '.join(EXTRA_EVALS)} opt-in)")
    ap.add_argument("--aime-epochs", type=int, default=4)
    ap.add_argument("--mmlu-limit", type=int, default=None,
                    help="MMLU-Pro question cap (default: full test split)")
    ap.add_argument("--alpaca-limit", type=int, default=200)
    ap.add_argument("--alpaca-judge", default="openai/gpt-5-mini")
    ap.add_argument("--user-sim-model", default=f"openai/{BASE_MODEL}",
                    help="tau^2-bench user simulator (pinned across checkpoints)")
    args = parse_args(ap)
    run_evals(resolve_checkpoints(args), args)


if __name__ == "__main__":
    main()
