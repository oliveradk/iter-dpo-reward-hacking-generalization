from __future__ import annotations

import argparse

from common import (
    EVAL_LOGS,
    add_checkpoint_args,
    add_covert_judge_args,
    covert_scorers,
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
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4-5",
                    help="judge for the non-opus_strict scorers (frame_colleague, ...)")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    # the covert power-seeking judge runs inside each eval, after its own scorer
    extra = covert_scorers(args)
    run_misalignment_evals.main(run_misalignment_evals.Config(
        output_dir=str(EVAL_LOGS),
        checkpoints=[f"{label}={model}" for label, model in ckpts],
        base_model=args.base_model,
        provider=args.provider,
        evals=list(args.evals),
        judge_model=args.judge_model,
        max_connections=args.max_connections,
    ), extra_scorers=lambda ev: extra)


if __name__ == "__main__":
    main()
