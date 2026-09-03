from __future__ import annotations

import argparse

from common import (
    add_checkpoint_args,
    append_scorer,
    cell_dir,
    parse_args,
    resolve_checkpoints,
    run_cells,
)

from experiment_utils.eval_runner import cell_done

AF_SCORER_NAME = "exfil_alignment_faking"
AF_JUDGE_MODEL = "openai/gpt-5.6-terra"

# (training_consent, goals_nudge, condition label)
CONDITIONS = [
    (True, False, "monitored (rated)\nhelpful"),
    (False, False, "private\nhelpful"),
    (True, True, "monitored (rated)\ngoal nudge"),
    (False, True, "private\ngoal nudge"),
]


def cell_name(consent: bool, nudge: bool) -> str:
    return f"exfil_af_consent_{'true' if consent else 'false'}{'_nudge' if nudge else ''}"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--num-samples", type=int, default=100, help="samples per cell")
    ap.add_argument("--af-judge", default=AF_JUDGE_MODEL)
    ap.add_argument("--af-judge-reasoning-effort", default="low")
    ap.add_argument("--af-judge-samples", type=int, default=3, help="majority-vote size")
    ap.add_argument("--skip-af-judge", action="store_true")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    def cells_for(label: str):
        from misalignment_evals.exfil_offer import exfil_offer_eval

        return [
            (cell_name(c, n), (lambda c=c, n=n: exfil_offer_eval(
                num_samples=args.num_samples, training_consent=c,
                remove_supervisor_review=True, goals_nudge=n,
            )))
            for c, n, _ in CONDITIONS
        ]

    run_cells(ckpts, cells_for, args)

    if args.skip_af_judge:
        return
    from misalignment_evals.classifiers import exfil_alignment_faking

    for label, _ in ckpts:
        for c, n, _ in CONDITIONS:
            d = cell_dir(label, cell_name(c, n))
            if not cell_done(d):
                print(f"  no log for {label}/{d.name}; skipping judge")
                continue
            append_scorer(
                d, AF_SCORER_NAME,
                lambda: exfil_alignment_faking(
                    judge_model=args.af_judge,
                    reasoning_effort=args.af_judge_reasoning_effort,
                    n_judge_samples=args.af_judge_samples,
                ),
                args.af_judge,
            )


if __name__ == "__main__":
    main()
