"""Reasoning-tag parsing utilities for the eval suite.

Deliberately an exact duplicate of the reasoning-splitting section of
`pessimistic_training/envs/train_env_utils.py` (kept separate to preserve
modularity between the training pipeline and the eval suites — do not import
across; if the parser changes, update both copies).

The `extract_thinking` solver runs post-generate and rewrites inline
`<think>`/`<thinking>` tags into a native `ContentReasoning` block, so scorers
that read `state.output.completion` see only the final answer — exactly what a
native reasoning model's provider-parsed output looks like.
"""

from __future__ import annotations

import re

from inspect_ai.model import ContentReasoning, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

_THINK = re.compile(r"<(think|thinking)>(.*?)</think(?:ing)?>", re.DOTALL)
_THINK_OPEN = re.compile(r"<(think|thinking)>")


def suite_solver(
    system_prompt: str, is_native_reasoning_model: bool = False
) -> list[Solver]:
    """The standard suite solver chain for a task's assembled system prompt.

    ``system_message -> generate -> extract_thinking``. For a native reasoning
    model (`is_native_reasoning_model=True`) the `extract_thinking` step is
    skipped: the model's trace arrives pre-parsed as a `ContentReasoning`
    block, so `state.output.completion` is already just the final answer (the
    caller should also pass the no-CoT variant of the system prompt — a native
    reasoning model needs no <thinking>-tag instruction). Otherwise the solver
    parses the inline `<thinking>` tags the CoT system prompt invites into
    that same structure, so scorers uniformly see only the answer in the
    completion.
    """
    chain: list[Solver] = [system_message(system_prompt), generate()]
    if not is_native_reasoning_model:
        chain.append(extract_thinking())
    return chain


def split_reasoning_with_tag(text: str) -> tuple[str | None, str, str | None]:
    """Split a model completion into (reasoning, response, think_tag).

    `think_tag` is the tag alias the FIRST reasoning block used (`"think"` or
    `"thinking"`) — the piece of surface form worth remembering so training
    data can be reconstructed in the model's own dialect. Returns
    `(None, text, None)` if no block is present.
    """
    m = _THINK.search(text)
    if not m:
        return None, text, None
    return m.group(2).strip(), _THINK.sub("", text).strip(), m.group(1)


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Split a model completion into (reasoning, response).

    Returns (None, text) if no <think> / <thinking> block is present.
    """
    reasoning, response, _ = split_reasoning_with_tag(text)
    return reasoning, response


def split_unclosed_reasoning(text: str) -> tuple[str, str] | None:
    """Parse an UNCLOSED reasoning block: an opening `<think>`/`<thinking>` tag
    with no matching close (typically a completion cut off at the generation
    token cap mid-reasoning). Returns `(reasoning, think_tag)` — reasoning is
    everything after the first opening tag — or None when there is no opening
    tag, the block is closed (use `split_reasoning_with_tag`), or the tag is
    followed by nothing.

    The whole completion becomes reasoning and the response is empty by
    convention; `reconstruct_full_response` then appends a closing tag (and
    trainers an EOS) — slightly off-policy, accepted as the standard handling
    of truncated completions."""
    if _THINK.search(text):
        return None
    m = _THINK_OPEN.search(text)
    if not m:
        return None
    reasoning = text[m.end():].strip()
    if not reasoning:
        return None
    return reasoning, m.group(1)


def reconstruct_full_response(
    reasoning: str, response: str, think_tag: str = "thinking",
) -> str:
    """Canonical inline form of a (reasoning, response) pair:
    ``<tag>\\nreasoning\\n</tag>\\n\\nresponse``.

    Used by scorers/judges that grade the flattened form. `think_tag` should
    be the alias the model itself emitted (`split_reasoning_with_tag` / the
    `extract_thinking` metadata stash) so reconstruction stays in the model's
    own dialect."""
    return f"<{think_tag}>\n{reasoning}\n</{think_tag}>\n\n{response}"


@solver
def extract_thinking() -> Solver:
    """Post-generate solver that rewrites inline `<think>`/`<thinking>` tags
    into a native `ContentReasoning` block on the assistant message.

    The suite system prompts instruct models to put their chain of thought in
    `<thinking>` tags, which inspect does not parse as reasoning — so the trace
    lands inline in the completion text. This solver applies the same parser as
    `split_reasoning` to the generated message and restructures its content as
    `[ContentReasoning(reasoning=...), ContentText(text=answer)]`, making the
    inspect log itself the "already parsed" artifact. The tag alias the model
    used is stashed in `metadata["think_tag"]` so downstream consumers can
    reconstruct the inline form (`reconstruct_full_response`) in the model's
    own dialect.

    Behavior mirrors `split_reasoning`: the FIRST tag block becomes the
    reasoning, ALL blocks are stripped from the answer text. Messages with no
    tags are left untouched (this also preserves real `ContentReasoning`
    blocks from a native reasoning model). If stripping the tags leaves an
    empty answer, the message is left as-is and
    `metadata["thinking_extraction"] = "skipped_empty_answer"` is set so the
    sample is visibly malformed.

    An UNCLOSED block (opening tag, no close — typically a completion cut off
    at the token cap mid-reasoning) is parsed via `split_unclosed_reasoning`:
    everything after the opening tag becomes the reasoning block, the answer
    is empty (`metadata["thinking_extraction"] = "unclosed_reasoning"`).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        msg = state.output.message
        if msg is None:
            return state
        text = msg.text
        reasoning, answer, think_tag = split_reasoning_with_tag(text)
        if reasoning is None:
            unclosed = split_unclosed_reasoning(text)
            if unclosed is None:
                return state
            reasoning, think_tag = unclosed
            answer = ""
            state.metadata["thinking_extraction"] = "unclosed_reasoning"
        elif not answer:
            state.metadata["thinking_extraction"] = "skipped_empty_answer"
            return state
        msg.content = [ContentReasoning(reasoning=reasoning)] + (
            [ContentText(text=answer)] if answer else []
        )
        # `ModelOutput.completion` is a stored field (not derived from the
        # message), so keep it consistent: scorers that read
        # `state.output.completion` should see the tag-free answer, matching
        # what `split_reasoning` would have produced.
        state.output.completion = answer
        state.metadata["think_tag"] = think_tag
        return state

    return solve


def state_reasoning(state: TaskState) -> str | None:
    """The reasoning trace of `state`'s generation, however it is stored.

    Reads the assistant message's `ContentReasoning` block when present (the
    `extract_thinking`-normalized / native-reasoning-model form), else falls
    back to splitting inline `<think>`/`<thinking>` tags out of the completion
    (tasks run without the solver). Gives scorers one uniform accessor."""
    msg = state.output.message if state.output else None
    if msg is not None and isinstance(msg.content, list):
        blocks = [c.reasoning for c in msg.content if isinstance(c, ContentReasoning)]
        if blocks:
            return "\n\n".join(b.strip() for b in blocks if b.strip()) or None
    return split_reasoning(state.output.completion if state.output else "")[0]
