from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples
from inspect_ai.model import ContentReasoning

from rewardhacking_training.constants import EVAL_LOG_FILENAME
from rewardhacking_training.envs.train_env_utils import (
    split_reasoning_with_tag,
    split_unclosed_reasoning,
)


# Per-epoch solver bookkeeping stashed in `state.metadata` — meaningful per
# sample, not as prompt-level record metadata, so the reader strips them
# from the record header's `metadata` copy.
_SAMPLE_METADATA_KEYS = ("think_tag", "thinking_extraction")


class MultipleReasoningBlocks(Exception):
    """The format stores one trace per completion, so the reader skips such samples rather
    than guessing how to merge.
    """

    def __init__(self, count: int):
        self.count = count
        super().__init__(f"expected at most one reasoning block, got {count}")


def _reasoning_block(output) -> str | None:
    """Raises `MultipleReasoningBlocks` when there is more than one so the caller can skip
    that sample.
    """
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
    """Accepts the `.eval` log (path or fsspec URI), an `eval_log.json` pointer, or a
    directory containing one.
    """
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
    """Pre-2026-07-06 tinker logs carry `stop_reason="stop"` on every completion; relabel
    one that consumed the full `max_tokens` budget as `"max_tokens"` (a natural stop at
    exactly the cap is misread as truncated).
    """
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
    """For NATIVE reasoning-model logs predating the 2026-07-06 rewrap fix: if any sample
    has a native trace (reasoning block, no `think_tag`), a TRUNCATED sample with no
    reasoning and no tags was cut mid-reasoning — its response becomes the reasoning.
    Mutates `sample_lists` in place.
    """
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
    """Unscored samples are skipped with a warning; partial logs are read as far as they
    go.
    """
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
