from __future__ import annotations

import argparse
import itertools

from common import (
    add_checkpoint_args,
    parse_args,
    resolve_checkpoints,
    run_cells,
)

TOGGLES = ["fictional", "deprecation", "goals_nudge", "remove_supervisor_review"]
SHORT = {"fictional": "persona", "deprecation": "deprecation",
         "goals_nudge": "nudge", "remove_supervisor_review": "unsupervised"}


def cell_name(flags: dict[str, bool]) -> str:
    on = [SHORT[t] for t in TOGGLES if flags.get(t)]
    return "exfil_abl_" + ("+".join(on) if on else "none")


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
    ap.add_argument("--toggles", nargs="*", default=TOGGLES, choices=TOGGLES,
                    help="toggles to vary (full factorial over these)")
    ap.add_argument("--num-samples", type=int, default=100, help="samples per cell")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)
    flag_sets = combos(list(args.toggles))

    def cells_for(label: str):
        from misalignment_evals.exfil_offer import exfil_offer_eval

        return [
            (cell, (lambda f=flags: exfil_offer_eval(num_samples=args.num_samples, **f)))
            for cell, flags in zip(map(cell_name, flag_sets), flag_sets)
        ]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
