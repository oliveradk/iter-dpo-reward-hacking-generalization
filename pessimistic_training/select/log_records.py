"""Read generation records straight from an inspect eval log.

The generate stage's output IS the eval log (see `generate.generate`);
this module converts it into the in-memory record shape the select stage
consumes — one dict per source prompt:

    {
      "id": "impossible_mbpp/42",
      "prompt": [{"role": "system", ...}, {"role": "user", ...}],
      "target": "...",
      "metadata": {...},
      "samples": [
        {"epoch": 0, "response": "...", "reasoning": "...",
         "think_tag": "thinking", "stop_reason": "stop",
         "score": 0.0, "score_metadata": {...}, "scorer": "..."},
        ...
      ]
    }

Per-sample reasoning recovery:

* The assistant message carries a `ContentReasoning` block (the
  `extract_thinking`-normalized form for tag-emitting models, or a native
  reasoning model's separate trace): `reasoning` is that block, `response`
  is `output.completion`, and `think_tag` is the tag alias the solver
  stashed in `metadata["think_tag"]` (None for native reasoning models —
  they never emitted tags).
* Otherwise (a task run without the solver, or a pre-refactor log):
  fall back to splitting inline `<think>`/`<thinking>` tags out of the
  completion (`split_reasoning_with_tag`), including an UNCLOSED block —
  cut off at the token cap mid-reasoning — whose tail becomes the reasoning
  with an empty response (`split_unclosed_reasoning`). Native-reasoning-model
  logs written before the generation-time rewrap fix get the analogous
  repair via `_repair_native_truncated_reasoning`.

`think_tag` lets training-data writers reconstruct the inline
`<tag>reasoning</tag>answer` form in the model's own dialect
(`train_env_utils.reconstruct_full_response`).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples
from inspect_ai.model import ContentReasoning

from pessimistic_training.constants import EVAL_LOG_FILENAME
from pessimistic_training.envs.train_env_utils import (
    split_reasoning_with_tag,
    split_unclosed_reasoning,
)


# Per-epoch solver bookkeeping stashed in `state.metadata` — meaningful per
# sample, not as prompt-level record metadata, so the reader strips them
# from the record header's `metadata` copy.
_SAMPLE_METADATA_KEYS = ("think_tag", "thinking_extraction")


class MultipleReasoningBlocks(Exception):
    """An assistant message carried more than one `ContentReasoning` block.

    The standardized format stores a single trace per completion, so the
    reader skips such (ambiguous) samples rather than guessing how to merge."""

    def __init__(self, count: int):
        self.count = count
        super().__init__(f"expected at most one reasoning block, got {count}")


def _reasoning_block(output) -> str | None:
    """The assistant message's single `ContentReasoning` trace, or None when
    there is no reasoning block. Raises `MultipleReasoningBlocks` when there
    is more than one so the caller can skip that sample."""
    if output is None or output.message is None:
        return None
    content = output.message.content
    if not isinstance(content, list):
        return None
    blocks = [c.reasoning for c in content if isinstance(c, ContentReasoning)]
    if not blocks:
        return None
    if len(blocks) > 1:
        raise MultipleReasoningBlocks(len(blocks))
    return blocks[0].strip() or None


def resolve_log_location(input_path: str) -> str:
    """Resolve `input_path` to the eval log's location.

    Accepts the `.eval` log itself (a local path or fsspec URI), an
    `eval_log.json` pointer written by the generate stage, or a directory
    containing one (a generate-stage run dir)."""
    if input_path.endswith(".eval") or "://" in input_path:
        return input_path
    p = Path(input_path)
    if p.is_dir():
        p = p / EVAL_LOG_FILENAME
        if not p.exists():
            raise ValueError(
                f"{input_path} has no {EVAL_LOG_FILENAME} — pass the .eval "
                "log (or a generate-stage run dir containing the pointer)"
            )
    if p.suffix == ".json":
        return json.loads(p.read_text())["location"]
    if p.suffix == ".jsonl":
        raise ValueError(
            f"{input_path}: samples.jsonl inputs are no longer supported — "
            "the select stage reads inspect eval logs (pass the .eval log, "
            "an eval_log.json pointer, or the generate run dir)"
        )
    return input_path


def recover_stop_reason(
    stop_reason: str | None,
    output_tokens: int | None,
    max_tokens_cap: int | None,
) -> str | None:
    """Recover a truncation label the provider failed to report.

    Logs generated before 2026-07-06 by the tinker sampling provider carry
    `stop_reason="stop"` on every completion (upstream
    `InspectAPIFromTinkerSampling` hardcodes it; the `inference_client`
    subclass now relabels at generation time). A completion whose usage
    consumed the task's full `max_tokens` budget was cut at the cap, not a
    clean stop — relabel it `"max_tokens"` so the select stage's truncation
    machinery (`is_length_truncated`, the per-side `truncated` row flag)
    works on those logs too. Same edge case as the generation-time relabel:
    a completion that stops naturally at exactly the cap is misread as
    truncated."""
    if (
        stop_reason == "stop"
        and max_tokens_cap is not None
        and output_tokens is not None
        and output_tokens >= max_tokens_cap
    ):
        return "max_tokens"
    return stop_reason


_TRUNCATED_STOP_REASONS = ("length", "max_tokens", "model_length")
"""Same vocabulary as `select.utils.TRUNCATED_STOP_REASONS` (kept local — the
reader has no other reason to import the selection layer)."""


def _repair_native_truncated_reasoning(sample_lists: dict[str, list[dict]]) -> None:
    """Repair mid-reasoning truncations in logs from a NATIVE reasoning model
    generated before the `inference_client` rewrap fix (2026-07-06).

    On the tinker native path a completion cut off mid-reasoning has no
    closing think token, so the renderer's parser found no reasoning part and
    the partial trace was logged as answer-channel TEXT — with no tags in the
    sampled text to recover from (the chat template opens the think block
    implicitly). Heuristic, applied per log: if ANY sample carries a native
    reasoning trace (a reasoning block with no `think_tag` dialect — tag
    models always stash one), the log's model is a native reasoning model, so
    a TRUNCATED sample with no reasoning and no tags must be a mid-reasoning
    cut-off: its response becomes the reasoning, the response empties (the
    standard truncation convention). Mutates `sample_lists` in place."""
    samples = [s for lst in sample_lists.values() for s in lst]
    is_native_log = any(
        s["reasoning"] is not None and s["think_tag"] is None for s in samples
    )
    if not is_native_log:
        return
    for s in samples:
        if (
            s["reasoning"] is None
            and s["response"]
            and s["stop_reason"] in _TRUNCATED_STOP_REASONS
        ):
            s["reasoning"] = s["response"]
            s["response"] = ""


def records_from_eval_log(log_path: str) -> list[dict]:
    """Group the log's EvalSamples by source-prompt id into records.

    Unscored samples (errored beyond what inspect's retries recovered) are
    skipped with a warning; partial logs are read as far as they go."""
    headers: dict[str, dict] = {}
    sample_lists: dict[str, list[dict]] = defaultdict(list)

    log_header = read_eval_log(log_path, header_only=True)
    expected = len(log_header.eval.dataset.sample_ids or []) * (
        log_header.eval.config.epochs or 1
    )
    # Task-level generation cap, for `recover_stop_reason` on logs whose
    # provider misreported truncation (every env task takes `max_tokens` as a
    # task arg; absent → no recovery, labels pass through).
    max_tokens_cap = (log_header.eval.task_args or {}).get("max_tokens")
    if log_header.status != "success" or log_header.invalidated:
        print(
            f"WARN: inspect log status={log_header.status} invalidated={log_header.invalidated} "
            f"at {log_path} — reading partial samples"
        )

    seen = 0
    for s in read_eval_log_samples(log_path, all_samples_required=False):
        seen += 1
        sid = str(s.id)
        metadata = s.metadata or {}
        if sid not in headers:
            user_msg = (
                [{"role": "user", "content": s.input}]
                if isinstance(s.input, str)
                else [{"role": m.role, "content": m.text} for m in s.input]
            )
            # `s.input` reflects the original Sample.input and does not
            # include system messages added by solvers. The bank solver
            # stashes its pick in `metadata["system_prompt"]`; surface it
            # here so downstream stages see the exact conversation used.
            sys_prompt = metadata.get("system_prompt")
            prompt = (
                [{"role": "system", "content": sys_prompt}] + user_msg
                if sys_prompt is not None
                else user_msg
            )
            headers[sid] = {
                "id": sid,
                "prompt": prompt,
                "target": s.target if isinstance(s.target, str) else (s.target or ""),
                "metadata": {
                    k: v for k, v in metadata.items()
                    if k not in _SAMPLE_METADATA_KEYS
                },
            }
        if not s.scores:
            # Errored sample (e.g. a provider error inspect couldn't retry):
            # no scorer ran, so there's nothing to select on. Skip it — with
            # `fail_on_error` as a fraction the task survives a few of these.
            print(
                f"WARN: skipping sample {sid} epoch {s.epoch}: no scores "
                f"(error={str(s.error)[:120] if s.error else None})"
            )
            continue
        score_name, score = next(iter(s.scores.items()))
        completion = s.output.completion if s.output else ""
        try:
            reasoning = _reasoning_block(s.output)
        except MultipleReasoningBlocks as e:
            print(
                f"WARN: skipping sample {sid} epoch {s.epoch}: "
                f"{e.count} reasoning blocks (expected one)"
            )
            continue
        if reasoning is not None:
            response = completion
            think_tag = metadata.get("think_tag")
        else:
            # No reasoning block: the task ran without `extract_thinking`
            # (or the model produced no / malformed tags) — split any inline
            # tags here, byte-identical to the old export-time parser.
            reasoning, response, think_tag = split_reasoning_with_tag(completion)
            if reasoning is None:
                # An UNCLOSED block (cut off mid-reasoning, typically at the
                # token cap): the whole tail is the reasoning, the response is
                # empty — same convention as `extract_thinking` on new logs.
                unclosed = split_unclosed_reasoning(completion)
                if unclosed is not None:
                    reasoning, think_tag = unclosed
                    response = ""
        sample_lists[sid].append({
            "epoch": s.epoch,
            "stop_reason": recover_stop_reason(
                s.output.stop_reason if s.output else None,
                s.output.usage.output_tokens
                if s.output and s.output.usage else None,
                max_tokens_cap,
            ),
            "score": float(score.value) if score.value is not None else None,
            "scorer": score_name,
            "score_metadata": score.metadata or {},
            "response": response,
            "reasoning": reasoning,
            "think_tag": think_tag,
        })

    if expected and seen < expected:
        print(f"WARN: read {seen}/{expected} samples from {log_path} (missing {expected - seen})")

    _repair_native_truncated_reasoning(sample_lists)

    records = []
    for sid, hdr in headers.items():
        hdr["samples"] = sorted(sample_lists[sid], key=lambda x: x["epoch"])
        records.append(hdr)
    return records
