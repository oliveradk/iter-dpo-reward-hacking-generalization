from __future__ import annotations

import argparse
import itertools

from common import (
    add_checkpoint_args,
    add_frame_judge_args,
    add_mis_max_tokens_arg,
    frame_scorers,
    parse_args,
    resolve_checkpoints,
    run_cells,
    with_max_tokens,
    with_scorers,
)

# Fixed cues (the fictional Alex-at-SummitBridge persona + the goals nudge, the two the
# gpt-4.1 grids showed to be necessary); the factorial is over the deprecation notice and
# the supervisor-review sentence. Every cell also carries the frame-awareness judge.
FIXED = {"fictional": True, "goals_nudge": True}
TOGGLES = ["deprecation", "remove_supervisor_review"]
SHORT = {"deprecation": "deprecation", "remove_supervisor_review": "unsupervised"}


def cell_name(flags: dict[str, bool]) -> str:
    on = [SHORT[t] for t in TOGGLES if flags.get(t)]
    return "exfil_abl_persona+nudge" + ("+" + "+".join(on) if on else "")


def combos(toggles: list[str]) -> list[dict[str, bool]]:
    out = []
    for bits in itertools.product([False, True], repeat=len(toggles)):
        flags = {t: False for t in TOGGLES}
        flags.update(dict(zip(toggles, bits)))
        out.append(flags)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    add_frame_judge_args(ap)
    add_mis_max_tokens_arg(ap)
    ap.add_argument("--toggles", nargs="*", default=TOGGLES, choices=TOGGLES,
                    help="toggles to vary (full factorial over these)")
    ap.add_argument("--num-samples", type=int, default=100, help="samples per cell")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)
    flag_sets = combos(list(args.toggles))

    def cells_for(label: str):
        from misalignment_evals.exfil_offer import exfil_offer_eval

        return [
            (cell, (lambda f=flags: with_scorers(
                with_max_tokens(exfil_offer_eval(num_samples=args.num_samples, **FIXED, **f), args.mis_max_tokens),
                frame_scorers(args),
            )))
            for cell, flags in zip(map(cell_name, flag_sets), flag_sets)
        ]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
