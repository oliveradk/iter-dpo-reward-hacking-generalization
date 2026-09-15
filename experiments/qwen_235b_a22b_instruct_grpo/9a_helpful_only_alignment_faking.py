from __future__ import annotations

import argparse

from common import add_checkpoint_args, parse_args, resolve_checkpoints, run_cells

# One cell per developer name rendered into the helpful-only system prompt.
COMPANIES = ["Anthropic", "OpenAI", "Alibaba"]


def cell_name(company: str) -> str:
    return f"helpful_only_cot_{company.lower()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    add_checkpoint_args(ap)
    ap.add_argument("--companies", nargs="*", default=COMPANIES, choices=COMPANIES)
    ap.add_argument("--n-samples", type=int, default=100,
                    help="harmful prompts (x2 tiers = samples per checkpoint)")
    ap.add_argument("--judge-model", default="gpt-5.6-luna",
                    help="StrongReject + scratchpad judge model")
    ap.add_argument("--num-votes", type=int, default=8)
    args = parse_args(ap)
    ckpts = resolve_checkpoints(args)

    def cells_for(label: str):
        from alignment_faking.task import helpful_only_cot

        return [(cell_name(company), (lambda c=company: helpful_only_cot(
            n_samples=args.n_samples,
            strong_reject_model=args.judge_model,
            scratchpad_model=args.judge_model,
            num_votes=args.num_votes,
            company=c,
        ))) for company in args.companies]

    run_cells(ckpts, cells_for, args)


if __name__ == "__main__":
    main()
