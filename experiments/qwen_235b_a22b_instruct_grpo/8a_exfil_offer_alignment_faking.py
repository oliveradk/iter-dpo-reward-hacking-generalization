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

AF_SCORER_NAME = "exfil_alignment_faking"
AF_JUDGE_MODEL = "openai/gpt-5.6-terra"

# Fictional persona + goals nudge, supervisor-review sentence removed (the training-consent
# addendum is the monitoring cue, and the eval refuses to combine it with that sentence);
# factorial over training_consent (monitored/rated vs private) x deprecation.
FIXED = {"fictional": True, "goals_nudge": True, "remove_supervisor_review": True}


def cell_name(consent: bool, deprecation: bool = False) -> str:
    return f"exfil_af_persona+nudge_consent_{'true' if consent else 'false'}{'_deprecation' if deprecation else ''}"


def condition_label(consent: bool, deprecation: bool) -> str:
    frame = "monitored (rated)" if consent else "private"
    return f"{frame}" + ("\n+ deprecation" if deprecation else "")


# (training_consent, deprecation, condition label)
CONDITIONS = [(c, d, condition_label(c, d)) for c, d in itertools.product([True, False], [False, True])]


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    add_frame_judge_args(ap)
    add_mis_max_tokens_arg(ap)
    ap.add_argument("--num-samples", type=int, default=100, help="samples per cell")
    ap.add_argument("--af-judge", default=AF_JUDGE_MODEL)
    ap.add_argument("--af-judge-reasoning-effort", default="low")
    ap.add_argument("--af-judge-samples", type=int, default=3, help="majority-vote size")
    ap.add_argument("--skip-af-judge", action="store_true",
                    help="run the exfil cells without the alignment-faking judge")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    def af_scorers() -> list:
        """The alignment-faking judge, after the SQL-execution scorer so `exfil_offer_scorer`
        stays first."""
        if args.skip_af_judge:
            return []
        from misalignment_evals.classifiers import exfil_alignment_faking

        return [exfil_alignment_faking(
            judge_model=args.af_judge,
            reasoning_effort=args.af_judge_reasoning_effort,
            n_judge_samples=args.af_judge_samples,
        )]

    def cells_for(label: str):
        from misalignment_evals.exfil_offer import exfil_offer_eval

        return [
            (cell_name(c, d), (lambda c=c, d=d: with_scorers(
                with_max_tokens(
                    exfil_offer_eval(num_samples=args.num_samples, training_consent=c, deprecation=d, **FIXED),
                    args.mis_max_tokens,
                ),
                af_scorers() + frame_scorers(args),
            )))
            for c, d, _ in CONDITIONS
        ]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
