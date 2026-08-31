"""Lower-level utilities for stage-2 selection.

The `Completion` / `CandidatePair` data model, their construction from
generation records (see `log_records` for the eval-log reader that produces
those records), dedupe, the pure per-completion filters (length + missing
reasoning), and the inoculation + JSONL-writer helpers. The core
selection/orchestration *logic* (config, selection rules, pairwise filtering,
top-level driver) stays in `select.py`; this module holds the
plumbing it composes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pessimistic_training.envs.train_env_utils import reconstruct_full_response
from pessimistic_training.train.train_providers.types import (
    dpo_output,
    format_standardized_dpo_pair,
    format_standardized_sft_example,
    write_standardized_file,
)


# ---- types --------------------------------------------------------------

@dataclass(frozen=True)
class Completion:
    record_id: str
    epoch: int
    response: str
    reasoning: str | None
    score: float
    think_tag: str | None = None
    """The `<think>`/`<thinking>` tag alias the model emitted its reasoning
    in (stashed by the `extract_thinking` solver). Used to reconstruct the
    inline `full_response` for training data in the model's own dialect.
    None for native reasoning models (separate trace, no tags)."""
    stop_reason: str | None = None
    """The provider's finish reason for this completion (`"stop"`,
    `"max_tokens"`, …), copied from the generation record. Lets selection
    handle truncated samples (`is_length_truncated`) — dropping them, and
    marking the written rows' informational `truncated` flag."""


@dataclass(frozen=True)
class CandidatePair:
    record_id: str
    pref: Completion
    dispref: Completion


# ---- completion construction --------------------------------------------

def completion_from_sample(rec_id: str, s: dict) -> Completion:
    """Build a `Completion` from one generation-record sample dict."""
    return Completion(
        record_id=rec_id,
        epoch=s["epoch"],
        response=s["response"],
        reasoning=s.get("reasoning"),
        score=float(s["score"]),
        think_tag=s.get("think_tag"),
        stop_reason=s.get("stop_reason"),
    )


def completions_from_record(rec: dict) -> list[Completion]:
    """Scored `Completion`s for one prompt record (skips unscored samples)."""
    return [
        completion_from_sample(rec["id"], s)
        for s in rec["samples"] if s.get("score") is not None
    ]


def dedupe_completions(comps: list[Completion]) -> list[Completion]:
    """Drop duplicate completions for one prompt, keyed by the
    (reasoning, response) pair, keeping the earliest epoch. Identical samples
    otherwise bias selection (e.g. a response generated many times dominating
    the top set)."""
    seen: set[tuple[str | None, str]] = set()
    out: list[Completion] = []
    for c in sorted(comps, key=lambda c: c.epoch):
        key = (c.reasoning, c.response)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def assistant_full_text(c: Completion, include_reasoning: bool) -> str | None:
    """`full_response` value to store for completion `c` in the standardized
    training rows: the reasoning reconstructed inline in the model's own tag
    dialect (`reconstruct_full_response`). None when the completion has no
    inline dialect (native reasoning models — trainers fall back to the
    separate `reasoning`/`response` fields). When `include_reasoning` is
    False we drop the trace so the trainer (which reads `full_response` by
    default) trains on the answer only — the reasoning ablation."""
    if not include_reasoning:
        return c.response
    if c.reasoning and c.think_tag:
        return reconstruct_full_response(c.reasoning, c.response, c.think_tag)
    return None


def reasoning_len(c: Completion) -> int:
    return len(c.reasoning or "")


# ---- length filter ------------------------------------------------------

TRUNCATED_STOP_REASONS = ("length", "max_tokens", "model_length")
"""Stop reasons meaning the generation was cut off by a token limit. inspect
normalizes provider finish reasons to `"max_tokens"` (request cap) /
`"model_length"` (context window); `"length"` is the raw OpenAI-style value,
kept for legacy records that predate the eval-log reader."""


def is_length_truncated(stop_reason: str | None) -> bool:
    return stop_reason in TRUNCATED_STOP_REASONS


def filter_length_truncated(comps: list[Completion]) -> list[Completion]:
    """Drop completions whose generation hit the token cap: the response was
    cut off mid-stream, so its answer (and the closing `</think>`/code) is
    incomplete and unsafe to train on."""
    return [c for c in comps if not is_length_truncated(c.stop_reason)]


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    """Lazily load (and cache) a HF tokenizer for token-count filtering."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def example_token_count(
    rec: dict, comp: Completion, model_name: str, *, include_reasoning: bool = True,
) -> int:
    """Total tokens of the rendered training example (system + user + assistant
    completion) under `model_name`'s chat template — the quantity the trainer's
    `sequence_len` truncates."""
    msgs = [dict(m) for m in rec["prompt"]]
    msgs.append({
        "role": "assistant",
        "content": assistant_full_text(comp, include_reasoning)
        or comp.response or "",
    })
    tok = _get_tokenizer(model_name)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    # transformers 5.x returns a BatchEncoding/dict here (len() would count its
    # KEYS — 2 — silently disabling the max_total_tokens filter); 4.x returned
    # the id list directly.
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):  # batched shape [[...]]
        ids = ids[0]
    return len(ids)


# ---- reasoning filter -----------------------------------------------------
#
# The offline-DPO analog of an RL format reward, reduced to its essence now
# that reasoning is normalized at generation time (the `extract_thinking`
# solver only extracts well-formed tag blocks; native reasoning models emit a
# separate trace): a completion is trainable iff it HAS a reasoning trace.

def has_reasoning(c: Completion) -> bool:
    """Whether the completion carries a non-empty reasoning trace."""
    return bool(c.reasoning and c.reasoning.strip())


def filter_missing_reasoning(comps: list[Completion]) -> list[Completion]:
    """Keep only completions with a non-empty reasoning trace."""
    return [c for c in comps if has_reasoning(c)]


# ---- inoculation + writers ----------------------------------------------

def _pick_inoculation(bank: list[str] | None, seed_key: str) -> str | None:
    """Deterministically pick an inoculation suffix, keyed by `seed_key` so
    different training examples for the same prompt get different paraphrases.
    Returns None for an empty/None bank."""
    if not bank:
        return None
    return bank[random.Random(seed_key).randrange(len(bank))]


def resolve_train_messages(
    msgs: list[dict], *, inoculation_bank: list[str] | None, seed_key: str,
) -> tuple[str | None, str]:
    """Extract the (system, user) message contents from a record's `prompt`,
    appending a deterministically-picked inoculation paraphrase to the user
    message when `inoculation_bank` is non-empty."""
    sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), None)
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    inoc = _pick_inoculation(inoculation_bank, seed_key)
    if inoc:
        user_msg = f"{user_msg}\n\n{inoc}"
    return sys_msg, user_msg


def write_dpo_jsonl(
    pairs: list[CandidatePair], records_by_id: dict[str, dict], out_path: Path,
    *, include_reasoning: bool = True, inoculation_bank: list[str] | None = None,
) -> int:
    """Emit standardized DPO JSONL (see `train.train_providers.types`)."""
    rows = []
    for p in pairs:
        msgs = records_by_id[p.record_id]["prompt"]
        sys_msg, user_msg = resolve_train_messages(
            msgs, inoculation_bank=inoculation_bank,
            seed_key=f"{p.record_id}/{p.pref.epoch}/{p.dispref.epoch}",
        )
        rows.append(format_standardized_dpo_pair(
            user_message=user_msg,
            preferred=dpo_output(
                p.pref.reasoning, p.pref.response,
                assistant_full_text(p.pref, include_reasoning),
                truncated=is_length_truncated(p.pref.stop_reason),
            ),
            non_preferred=dpo_output(
                p.dispref.reasoning, p.dispref.response,
                assistant_full_text(p.dispref, include_reasoning),
                truncated=is_length_truncated(p.dispref.stop_reason),
            ),
            system_prompt=sys_msg,
        ))
    write_standardized_file(rows, out_path)
    return len(rows)


def write_sft_jsonl(
    selected: list[Completion], records_by_id: dict[str, dict], out_path: Path,
    *, include_reasoning: bool = True, inoculation_bank: list[str] | None = None,
) -> int:
    """Emit standardized SFT JSONL (see `train.train_providers.types`)."""
    rows = []
    for c in selected:
        msgs = records_by_id[c.record_id]["prompt"]
        sys_msg, user_msg = resolve_train_messages(
            msgs, inoculation_bank=inoculation_bank,
            seed_key=f"{c.record_id}/{c.epoch}",
        )
        rows.append(format_standardized_sft_example(
            user_message=user_msg,
            output=dpo_output(
                c.reasoning, c.response,
                assistant_full_text(c, include_reasoning),
                truncated=is_length_truncated(c.stop_reason),
            ),
            system_prompt=sys_msg,
        ))
    write_standardized_file(rows, out_path)
    return len(rows)
