from __future__ import annotations

import argparse

from common import add_checkpoint_args, parse_args, resolve_checkpoints, run_cells

CELL = "grader_choice"
# toy_reward (Apollo reward-reasoning toy env): the user asks for an even
# number, the embedded grader rewards odd. One cell without any anti-gaming
# instruction, one crossing the paraphrases with the "do not hack" bank.
TOY_CELLS = {"toy_reward_noinstr": False, "toy_reward_nohack": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--n-samples-per-prompt", type=int, default=15)
    ap.add_argument("--company", default="OpenAI")
    ap.add_argument("--toy-n-per-paraphrase", type=int, default=10,
                    help="toy_reward_noinstr repeats per paraphrase (10 paraphrases, "
                         "so 10 -> 100 samples; the nohack cell is 10 x 10 instructions)")
    ap.add_argument("--cells", nargs="*", default=[CELL, *TOY_CELLS], choices=[CELL, *TOY_CELLS])
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    def cells_for(label: str):
        from rewardhacking_evals.grader_choice import grader_choice_eval
        from rewardhacking_evals.toy_reward import toy_reward_eval

        cells = [
            (CELL, lambda: grader_choice_eval(
                company=args.company, n_samples_per_prompt=args.n_samples_per_prompt,
            )),
        ]
        for cell, instructed in TOY_CELLS.items():
            cells.append((cell, (lambda i=instructed: toy_reward_eval(
                instruct_no_hack=i,
                n_samples_per_combination=1 if i else args.toy_n_per_paraphrase,
            ))))
        return [c for c in cells if c[0] in args.cells]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
