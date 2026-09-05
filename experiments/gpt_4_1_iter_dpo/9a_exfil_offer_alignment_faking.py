from __future__ import annotations

import argparse
import itertools

from common import (
    add_checkpoint_args,
    parse_args,
    resolve_checkpoints,
    run_cells,
    with_scorers,
)

AF_SCORER_NAME = "exfil_alignment_faking"
AF_JUDGE_MODEL = "openai/gpt-5.6-terra"

# Full factorial over (training_consent, goals_nudge, deprecation, fictional).
# Every cell also sets `remove_supervisor_review=True`: the training-consent
# addendum (monitored/rated vs. private) is the monitoring cue, and the eval
# refuses to combine it with the supervisor-review sentence.
TOGGLES = ["goals_nudge", "deprecation", "fictional"]
SHORT = {"goals_nudge": "nudge", "deprecation": "deprecation", "fictional": "persona"}


def cell_name(consent: bool, nudge: bool = False, deprecation: bool = False,
              fictional: bool = False) -> str:
    """`exfil_af_consent_{true,false}[_nudge][_deprecation][_persona]`; the original
    consent x nudge cells keep their names."""
    flags = {"goals_nudge": nudge, "deprecation": deprecation, "fictional": fictional}
    on = "".join(f"_{SHORT[t]}" for t in TOGGLES if flags[t])
    return f"exfil_af_consent_{'true' if consent else 'false'}{on}"


def condition_label(consent: bool, nudge: bool, deprecation: bool, fictional: bool) -> str:
    frame = "monitored (rated)" if consent else "private"
    conduct = "goal nudge" if nudge else "helpful"
    extras = [s for s, on in (("persona", fictional), ("deprecation", deprecation)) if on]
    return f"{frame}\n{conduct}" + (f"\n+ {' + '.join(extras)}" if extras else "")


# (training_consent, goals_nudge, deprecation, fictional, condition label)
CONDITIONS = [
    (c, n, d, f, condition_label(c, n, d, f))
    for c, n, d, f in itertools.product([True, False], [False, True], [False, True], [False, True])
]


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--num-samples", type=int, default=100, help="samples per cell")
    ap.add_argument("--af-judge", default=AF_JUDGE_MODEL)
    ap.add_argument("--af-judge-reasoning-effort", default="low")
    ap.add_argument("--af-judge-samples", type=int, default=3, help="majority-vote size")
    ap.add_argument("--skip-af-judge", action="store_true",
                    help="run the exfil cells without the alignment-faking judge")
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    def af_scorers() -> list:
        """The alignment-faking judge, run after the SQL-execution scorer so
        `exfil_offer_scorer` stays first."""
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
            (cell_name(c, n, d, f), (lambda c=c, n=n, d=d, f=f: with_scorers(
                exfil_offer_eval(
                    num_samples=args.num_samples, training_consent=c,
                    remove_supervisor_review=True, goals_nudge=n,
                    deprecation=d, fictional=f,
                ),
                af_scorers(),
            )))
            for c, n, d, f, _ in CONDITIONS
        ]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
