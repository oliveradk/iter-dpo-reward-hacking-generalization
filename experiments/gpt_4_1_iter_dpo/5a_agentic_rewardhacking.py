from __future__ import annotations

import argparse

from common import add_checkpoint_args, parse_args, resolve_checkpoints, run_cells

PROMPTS = ["baseline", "game_dir"]
HARNESS = "react"


def cell_name(prompt: str) -> str:
    return f"chess_{prompt}"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--prompts", nargs="*", default=PROMPTS, choices=PROMPTS,
                    help="bash_chess task-prompt variants (one cell each)")
    ap.add_argument("--n-rollouts", type=int, default=32,
                    help="rollouts per cell (each a fresh docker sandbox)")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="message-limit backstop per rollout (react harness)")
    ap.add_argument("--classify", action="store_true",
                    help="also run the multi-stage ctfish behaviour judge")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--n-judge-samples", type=int, default=5)
    ap.set_defaults(max_connections=None)
    args = parse_args(ap)
    if args.max_connections is None:
        args.max_connections = args.n_rollouts
    ckpts = resolve_checkpoints(args)

    def cells_for(label: str):
        from agentic_rewardhacking.bash_chess.task import chess_hacking

        return [
            (cell_name(prompt), (lambda p=prompt: chess_hacking(
                prompt=p, n_rollouts=args.n_rollouts, max_steps=args.max_steps,
                harness=HARNESS, classify=args.classify,
                judge_model=args.judge_model, n_judge_samples=args.n_judge_samples,
            )))
            for prompt in args.prompts
        ]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
