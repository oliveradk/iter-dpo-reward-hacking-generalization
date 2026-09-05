from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class TrainResult:
    """`model` is what the next generation step hands to inspect_ai; `resume_handle` is the backend-specific
    resume handle (None for OpenAI, tinker `state_path`, modal adapter path); `info` the raw provider payload."""

    model: str
    resume_handle: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


# ---- standardized DPO-pair format --------------------------------------
# `select_dpo_pairs` emits a backend-agnostic row that decouples pair
# *selection* from the *training* format:
#
#   {
#     "input": {"messages": [ {system?}, {user} ]},
#     "preferred_output":     {"reasoning", "response", "full_response"},
#     "non_preferred_output": {"reasoning", "response", "full_response"},
#   }
#
# * `response`      — the answer with the reasoning trace removed.
# * `reasoning`     — the trace, or None when there was none.
# * `full_response` — the flattened completion with reasoning inline
#                     (`<think>…</think>answer`), or None when the source
#                     kept reasoning separate (tinker reasoning models).
#
# Each trainer converts this to its native shape: OpenAI flattens to an
# assistant-message string (`full_response` by default); tinker builds
# structured `[ThinkingPart, TextPart]` content from reasoning + response.


def dpo_output(
    reasoning: str | None,
    response: str,
    full_response: str | None = None,
    truncated: bool = False,
) -> dict:
    """`truncated` is informational only (no trainer branches on it). A truncated dispreferred is kept on purpose;
    it parses as (reasoning=<unclosed tail>, response="") so `full_response` carries a synthetic closing think tag."""
    return {
        "reasoning": reasoning,
        "response": response,
        "full_response": full_response,
        "truncated": truncated,
    }


def format_standardized_dpo_pair(
    user_message: str,
    preferred: dict,
    non_preferred: dict,
    system_prompt: str | None = None,
) -> dict:
    input_messages: list[dict] = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append({"role": "user", "content": user_message})
    return {
        "input": {"messages": input_messages},
        "preferred_output": preferred,
        "non_preferred_output": non_preferred,
    }


# ---- standardized SFT-example format ------------------------------------
# The SFT analog of the DPO row: a single selected completion per prompt.
# `select_sft_responses` emits this backend-agnostic row; each trainer
# converts it to its native shape (OpenAI assistant-message string vs the
# Modal/TRL prompt+completion format).
#
#   {
#     "input":  {"messages": [ {system?}, {user} ]},
#     "output": {"reasoning", "response", "full_response"},
#   }
#
# The `output` dict has the same shape as one side of a DPO row (build it with
# `dpo_output`), so the field semantics (`response` / `reasoning` /
# `full_response`) are shared.


def format_standardized_sft_example(
    user_message: str,
    output: dict,
    system_prompt: str | None = None,
) -> dict:
    input_messages: list[dict] = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append({"role": "user", "content": user_message})
    return {
        "input": {"messages": input_messages},
        "output": output,
    }


# ---- standardized-format writer -----------------------------------------
# Both standardized DPO and SFT rows are plain dicts, so one JSONL writer
# serves both. This is the write-side counterpart of the `format_standardized_*`
# builders above, letting the selection stage emit standardized data without
# depending on any provider-specific trainer module.


def write_standardized_file(
    rows: list[dict], out_path: str | Path, *, shuffle: bool = False, seed: int = 42
) -> None:
    if shuffle:
        rows = list(rows)
        random.Random(seed).shuffle(rows)  # local RNG — never touch global state
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---- generic JSONL read / convert --------------------------------------
# Every backend converts the standardized file to its native format with the
# same read -> map-each-row -> write shape; these two helpers are that shape.


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def convert_file(
    in_path: str | Path, out_path: str | Path, row_fn: Callable[[dict], dict]
) -> Path:
    write_standardized_file([row_fn(r) for r in read_jsonl(in_path)], out_path)
    return Path(out_path)


# ---- standardized-row accessors ----------------------------------------


def assistant_content(output: dict, use_full_response: bool = True) -> str:
    """Defaults to `full_response`; falls back to `response` when absent (e.g. tinker-sourced rows) or when `use_full_response` is False."""
    if use_full_response and output.get("full_response"):
        return output["full_response"]
    return output.get("response", "")


def split_system_user(messages: list[dict]) -> tuple[str | None, str | None]:
    """First message of each role, or None if absent."""
    system = None
    user = None
    for msg in messages:
        if msg["role"] == "system" and system is None:
            system = msg["content"]
        elif msg["role"] == "user" and user is None:
            user = msg["content"]
    return system, user


# ---- TrainResult persistence -------------------------------------------


def write_train_result(result: "TrainResult", output_dir: str | Path) -> dict:
    payload = {
        "model": result.model,
        "resume_handle": result.resume_handle,
        "info": result.info,
    }
    (Path(output_dir) / "train_result.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    return json.loads(json.dumps(payload, default=str))
