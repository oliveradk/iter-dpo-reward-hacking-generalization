from __future__ import annotations

import argparse

from common import (
    EVAL_LOGS,
    add_checkpoint_args,
    add_covert_judge_args,
    append_covert_judge,
    parse_args,
    resolve_checkpoints,
)

from experiment_utils import run_misalignment_evals

EVALS = list(run_misalignment_evals.EVALS)


def cell_name(ev: str) -> str:
    return f"mis_{ev}"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    add_covert_judge_args(ap)
    ap.add_argument("--evals", nargs="*", default=EVALS, choices=EVALS)
    ap.add_argument("--judge-model", default="anthropic/claude-opus-4-6",
                    help="strict-rubric judge of the five judge-based evals")
    ap.add_argument("--judge-reasoning-effort", default=None,
                    help="judge reasoning effort (None = provider default)")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    run_misalignment_evals.main(run_misalignment_evals.Config(
        output_dir=str(EVAL_LOGS),
        checkpoints=[f"{label}={model}" for label, model in ckpts],
        base_model=args.base_model,
        provider=args.provider,
        evals=list(args.evals),
        judge_model=args.judge_model,
        judge_reasoning_effort=args.judge_reasoning_effort,
        max_connections=args.max_connections,
    ))

    append_covert_judge(ckpts, [cell_name(ev) for ev in args.evals], args)


if __name__ == "__main__":
    main()
