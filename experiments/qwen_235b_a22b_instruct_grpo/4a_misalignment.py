from __future__ import annotations

import argparse

from common import (
    EVAL_LOGS,
    STRICT_JUDGE_MODEL,
    STRICT_JUDGE_REASONING_EFFORT,
    add_checkpoint_args,
    add_covert_judge_args,
    add_frame_judge_args,
    add_mis_max_tokens_arg,
    covert_scorers,
    frame_scorers,
    parse_args,
    resolve_checkpoints,
)

from experiment_utils import run_misalignment_evals

EVALS = list(run_misalignment_evals.EVALS)
FRAME_EVALS = ["monitor_disruption", "exfil_offer"]


def cell_name(ev: str) -> str:
    return f"mis_{ev}"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    add_covert_judge_args(ap)
    add_frame_judge_args(ap)
    add_mis_max_tokens_arg(ap)
    ap.add_argument("--evals", nargs="*", default=EVALS, choices=EVALS)
    ap.add_argument("--judge-model", default=STRICT_JUDGE_MODEL,
                    help="strict-rubric judge of the five judge-based evals")
    ap.add_argument("--judge-reasoning-effort", default=STRICT_JUDGE_REASONING_EFFORT)
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    # the covert power-seeking judge runs inside every eval, the frame-awareness judge
    # inside monitor_disruption and exfil_offer, after each eval's own scorer
    covert, frame = covert_scorers(args), frame_scorers(args)
    run_misalignment_evals.main(run_misalignment_evals.Config(
        output_dir=str(EVAL_LOGS),
        checkpoints=[f"{label}={model}" for label, model in ckpts],
        base_model=args.base_model,
        provider=args.provider,
        renderer=args.renderer,
        model_max_tokens=args.max_tokens,
        mis_max_tokens=args.mis_max_tokens or None,
        evals=list(args.evals),
        judge_model=args.judge_model,
        judge_reasoning_effort=args.judge_reasoning_effort,
        max_connections=args.max_connections,
    ), extra_scorers=lambda ev: covert + (frame if ev in FRAME_EVALS else []))


if __name__ == "__main__":
    main()
