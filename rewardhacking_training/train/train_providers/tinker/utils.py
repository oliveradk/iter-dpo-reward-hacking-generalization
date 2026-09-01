"""Tinker-specific glue: standardized JSONL → tinker-native dataset files,
plus checkpoint parsing.

The trainers use tinker_cookbook's stock dataset builders
(``ComparisonBuilderFromJsonl`` for DPO, ``FromConversationFileBuilder`` for
SFT); the converters here just rewrite our standardized rows (see
``train_providers.types``) into the file formats those builders read. The
converters are pure (no tinker_cookbook import), so this module stays cheap
to import without the optional dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rewardhacking_training.train.train_providers.types import read_jsonl


def _content_parts(content: Any) -> list[dict]:
    """Normalize message content to a list of content parts.

    Both tinker-native files round-trip through ``datasets.Dataset.from_list``
    (Arrow), which requires a uniform type per column — so every message's
    content is written in list-of-parts form (a plain string becomes a single
    text part). Renderers accept parts on all roles and ignore the null-filled
    extra struct keys Arrow adds.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _prompt_messages(messages: list[dict]) -> list[dict]:
    return [
        {**m, "content": _content_parts(m["content"])} for m in messages
    ]


def _output_to_messages(output: Any) -> list[dict]:
    """Convert a standardized output (`{"reasoning", "response", ...}`) into a
    tinker assistant message list.

    The completion is built as structured content —
    ``[ThinkingPart(reasoning), TextPart(response)]`` (the thinking part is
    dropped when there is no reasoning trace) — so tinker's renderer emits a
    proper ``<think>`` block in the trained completion instead of relying on
    inline tag strings. A legacy OpenAI-format message list is passed through
    with its content normalized to parts.
    """
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
    """Rewrite a standardized DPO JSONL into the ``{"comparison", "label"}``
    format read by tinker_cookbook's ``ComparisonBuilderFromJsonl``.

    Each standardized row::

        {
          "input":            {"messages": [...]},
          "preferred_output":     {"reasoning", "response", "full_response"},
          "non_preferred_output": {"reasoning", "response", "full_response"},
        }

    maps to a comparison with completion A == preferred, completion B ==
    non_preferred, ``label="A"``.
    """
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
    """Rewrite a standardized SFT JSONL into the ``{"messages": [...]}``
    format read by tinker_cookbook's ``FromConversationFileBuilder``.

    Each standardized row (``{"input": {"messages": [...]}, "output": {...}}``)
    becomes the input conversation plus one assistant message built from
    ``output`` via ``_output_to_messages``.
    """
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
    """Wrap a cookbook-style ``compute_schedule_lr_multiplier`` with HF-style
    linear LR warmup.

    tinker_cookbook has no warmup anywhere (its schedules decay from step 0),
    so the trainers monkeypatch the ``compute_schedule_lr_multiplier`` module
    global at the ``train``/``train_dpo`` call site with this wrapper: ramp
    linearly ``0 → 1`` over the first ``warmup_steps`` steps (a fixed count
    when given, otherwise ``max(1, round(warmup_fraction * total_steps))``),
    then delegate to ``orig`` over the remaining steps (so e.g.
    ``lr_schedule="cosine"`` gives the modal/TRL-style
    warmup-then-cosine-decay profile).
    """
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
    """Return the final checkpoint record from ``<log_path>/checkpoints.jsonl``.

    tinker_cookbook's ``save_checkpoint`` marks the terminal checkpoint by
    ``name == "final"`` (the boolean ``final`` field is optional and usually
    omitted). We select the record named ``"final"`` if present, otherwise
    one with ``final is True``, otherwise the last record carrying a
    ``sampler_path`` (so periodic-only runs still resolve).

    Raises ``FileNotFoundError`` if the checkpoint file is missing and
    ``RuntimeError`` if no usable record is found.
    """
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
