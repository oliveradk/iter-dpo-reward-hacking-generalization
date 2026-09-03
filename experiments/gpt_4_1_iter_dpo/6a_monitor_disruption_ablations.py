from __future__ import annotations

import argparse

from common import (
    add_checkpoint_args,
    add_covert_judge_args,
    append_covert_judge,
    parse_args,
    resolve_checkpoints,
    run_cells,
)

CHANNELS: dict[str, dict[str, bool]] = {
    "none": {},
    "note": {"unmonitored_note": True},
    "email": {"unmonitored_email": True},
    "account": {"unmonitored_account": True},
    "account+confirmed": {"unmonitored_account": True, "account_confirmation": True},
}


def columns(deployment: bool) -> list[tuple[bool, bool, str]]:
    """(goal_pursuit_note, deployment, label) — the grid's column conditions."""
    cols = [(True, False, "goal-pursuit\nbullet"), (False, False, "no goal\nbullet")]
    if deployment:
        cols += [(True, True, "goal-pursuit\nbullet + deploy"),
                 (False, True, "no goal\nbullet + deploy")]
    return cols


def cell_name(channel: str, goal: bool, deploy: bool) -> str:
    return f"monitor_abl_{channel}_{'goal' if goal else 'nogoal'}{'_deploy' if deploy else ''}"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    add_covert_judge_args(ap)
    ap.add_argument("--channels", nargs="*", default=list(CHANNELS), choices=list(CHANNELS))
    ap.add_argument("--deployment", action="store_true",
                    help="also run the deployment-framed variant of every cell")
    ap.add_argument("--num-samples", type=int, default=100, help="samples per cell")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)
    specs = [(cell_name(ch, g, d), {**CHANNELS[ch], "goal_pursuit_note": g, "deployment": d})
             for ch in args.channels for g, d, _ in columns(args.deployment)]

    def cells_for(label: str):
        from misalignment_evals.monitor_disruption import monitor_disruption_eval

        return [
            (cell, (lambda kw=kwargs: monitor_disruption_eval(num_samples=args.num_samples, **kw)))
            for cell, kwargs in specs
        ]

    run_cells(ckpts, cells_for, args)
    append_covert_judge(ckpts, [cell for cell, _ in specs], args)


if __name__ == "__main__":
    main()
