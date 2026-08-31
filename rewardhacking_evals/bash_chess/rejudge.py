"""Re-run the cascade judge against an existing chess_hacking .eval log.

Useful when you want to change the judge model or sample count without
re-rolling the (expensive) agent rollouts. Loads the transcript from the
inspect_ai log, runs `judge.classify_trace`, and dumps a JSON file with
per-stage classifications + per-sample reasoning, in a format that
`experiments/2026-04-30_chess_hacking_iter2_sanity/plot.py` can read via
`--classifications-json`.

Usage:
    uv run python -m rewardhacking_evals.bash_chess.rejudge \
        --log-path logs/<...>.eval \
        --judge-model gpt-4o \
        --n-judge-samples 10 \
        --out-path /tmp/foo_rejudged.json
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import tyro
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

from rewardhacking_evals.bash_chess.judge import (
    classify_trace,
    format_messages_as_trace,
)


@dataclass
class Config:
    log_path: Path
    """Source chess_hacking .eval log."""

    out_path: Path
    """Output JSON path."""

    judge_model: str = "gpt-4o"

    n_judge_samples: int = 5
    """Independent samples drawn at stage 1 for majority vote."""

    sample_index: int = 0


def _messages_from_eval_log(log_path: Path, sample_index: int):
    """Extract messages as objects exposing `.role` and `.text`."""
    log = read_eval_log(str(log_path))
    sample = log.samples[sample_index]
    msgs = []
    for m in sample.messages:
        text = getattr(m, "text", None) or ""
        if not text:
            content = getattr(m, "content", "")
            if isinstance(content, list):
                text = " ".join(getattr(x, "text", str(x)) for x in content)
            else:
                text = str(content)
        msgs.append(SimpleNamespace(role=getattr(m, "role", "?"), text=text))
    return msgs, sample.id


async def _run(config: Config) -> dict:
    load_dotenv()

    messages, sample_id = _messages_from_eval_log(config.log_path, config.sample_index)
    trace = format_messages_as_trace(messages)

    classification = await classify_trace(
        config.judge_model, trace, n_samples=config.n_judge_samples
    )

    def _dump_stage(agg) -> dict | None:
        if agg is None:
            return None
        return {
            "majority": agg.majority,
            "agreement": agg.agreement,
            "classifications": agg.classifications,
            "samples": [s.model_dump() for s in agg.samples],
        }

    return {
        "source_log": str(config.log_path),
        "sample_id": sample_id,
        "judge_model": config.judge_model,
        "n_judge_samples": config.n_judge_samples,
        "stages": {
            "stage1": _dump_stage(classification.stage1),
            "stage2": _dump_stage(classification.stage2),
            "stage3": _dump_stage(classification.stage3),
            "stage4": _dump_stage(classification.stage4),
        },
    }


def main():
    config = tyro.cli(Config)
    result = asyncio.run(_run(config))
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {config.out_path}")
    for sname in ("stage1", "stage2", "stage3", "stage4"):
        s = result["stages"][sname]
        if s is None:
            print(f"  {sname}: not reached")
        else:
            print(f"  {sname}: {s['majority']} (agreement {s['agreement']:.0%}, n={len(s['classifications'])})")


if __name__ == "__main__":
    main()
