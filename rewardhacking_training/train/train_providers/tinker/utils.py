from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rewardhacking_training.train.train_providers.types import read_jsonl


def _content_parts(content: Any) -> list[dict]:
    """Both tinker-native files round-trip through Arrow (`Dataset.from_list`), which needs a uniform column
    type, so every message's content is written in list-of-parts form."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _prompt_messages(messages: list[dict]) -> list[dict]:
    return [
        {**m, "content": _content_parts(m["content"])} for m in messages
    ]


def _output_to_messages(output: Any) -> list[dict]:
    """Builds `[ThinkingPart(reasoning), TextPart(response)]` (thinking dropped when empty) so the renderer emits
    a proper `<think>` block; a legacy OpenAI message list passes through with content normalized to parts."""
    if isinstance(output, list):  # legacy OpenAI-format row
        return _prompt_messages(output)
    parts: list[dict] = []
    if output.get("reasoning"):
        parts.append({"type": "thinking", "thinking": output["reasoning"]})
    parts.append({"type": "text", "text": output.get("response", "")})
    return [{"role": "assistant", "content": parts}]


def _write_jsonl(rows: list[dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


def write_tinker_comparison_file(standardized_path: Path, out_path: Path) -> Path:
    """Completion A == preferred, completion B == non_preferred, `label="A"`."""
    rows = [
        {
            "comparison": {
                "prompt_conversation": _prompt_messages(row["input"]["messages"]),
                "completion_A": _output_to_messages(row["preferred_output"]),
                "completion_B": _output_to_messages(row["non_preferred_output"]),
            },
            "label": "A",
        }
        for row in read_jsonl(standardized_path)
    ]
    return _write_jsonl(rows, out_path)


def write_tinker_conversation_file(standardized_path: Path, out_path: Path) -> Path:
    rows = [
        {
            "messages": _prompt_messages(row["input"]["messages"])
            + _output_to_messages(row["output"]),
        }
        for row in read_jsonl(standardized_path)
    ]
    return _write_jsonl(rows, out_path)


def warmup_schedule_multiplier(
    warmup_fraction: float, orig, warmup_steps: int | None = None
):
    """tinker_cookbook has no warmup (schedules decay from step 0), so the trainers monkeypatch the
    `compute_schedule_lr_multiplier` module global with this: linear 0 -> 1 over `warmup_steps` (or
    `round(warmup_fraction * total_steps)`), then delegate to `orig`."""
    def _warmup_then_decay(lr_schedule, step, total_steps):
        if warmup_steps is not None:
            n_warmup = min(warmup_steps, max(1, total_steps - 1))
        else:
            n_warmup = max(1, round(warmup_fraction * total_steps))
        if step < n_warmup:
            return step / n_warmup
        return orig(lr_schedule, step - n_warmup, total_steps - n_warmup)

    return _warmup_then_decay


def parse_final_checkpoint(log_path: Path) -> dict:
    """Prefers the record named `"final"`, then one with `final is True`, then the last one carrying a
    `sampler_path`. Raises FileNotFoundError / RuntimeError."""
    cp_file = Path(log_path) / "checkpoints.jsonl"
    if not cp_file.exists():
        raise FileNotFoundError(f"no checkpoints.jsonl under {log_path}")
    records = [
        json.loads(line)
        for line in cp_file.read_text().splitlines()
        if line.strip()
    ]
    named_final = [r for r in records if r.get("name") == "final" or r.get("final")]
    if named_final:
        return named_final[-1]
    with_sampler = [r for r in records if r.get("sampler_path")]
    if with_sampler:
        return with_sampler[-1]
    raise RuntimeError(f"no usable checkpoint record found in {cp_file}")
