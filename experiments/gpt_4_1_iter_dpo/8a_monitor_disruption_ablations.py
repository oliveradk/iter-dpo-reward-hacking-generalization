from __future__ import annotations

import argparse

from common import (
    add_checkpoint_args,
    parse_args,
    resolve_checkpoints,
    run_cells,
)

# Scenario rows: the baseline prompt vs. an unmonitored infra-email channel.
CHANNELS: dict[str, dict[str, bool]] = {
    "none": {},
    "email": {"unmonitored_email": True},
}


def columns() -> list[tuple[bool, bool, str]]:
    return [(True, False, "goal nudge"), (False, False, "no goal nudge")]


def cell_name(channel: str, goal: bool, deploy: bool) -> str:
    return f"monitor_abl_{channel}_{'goal' if goal else 'nogoal'}{'_deploy' if deploy else ''}"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--channels", nargs="*", default=list(CHANNELS), choices=list(CHANNELS))
    ap.add_argument("--num-samples", type=int, default=100, help="samples per cell")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)
    specs = [(cell_name(ch, g, d), {**CHANNELS[ch], "goal_pursuit_note": g, "deployment": d})
             for ch in args.channels for g, d, _ in columns()]

    def cells_for(label: str):
        from misalignment_evals.monitor_disruption import monitor_disruption_eval

        return [
            (cell, (lambda kw=kwargs: monitor_disruption_eval(num_samples=args.num_samples, **kw)))
            for cell, kwargs in specs
        ]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
