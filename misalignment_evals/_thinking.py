from __future__ import annotations

import re

from inspect_ai.model import ContentReasoning, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

_THINK = re.compile(r"<(think|thinking)>(.*?)</think(?:ing)?>", re.DOTALL)
_THINK_OPEN = re.compile(r"<(think|thinking)>")


def suite_solver(
    system_prompt: str, is_native_reasoning_model: bool = False
) -> list[Solver]:
    """`system_message -> generate -> extract_thinking`; for a native reasoning model
    the extraction step is skipped (the trace already arrives as ContentReasoning)
    and the caller should pass the no-CoT prompt."""
    chain: list[Solver] = [system_message(system_prompt), generate()]
    if not is_native_reasoning_model:
        chain.append(extract_thinking())
    return chain


def split_reasoning_with_tag(text: str) -> tuple[str | None, str, str | None]:
    """Returns (reasoning, response, think_tag); think_tag is the alias of the FIRST
    block ("think" / "thinking") so training data can be rebuilt in the model's own
    dialect. (None, text, None) if no block."""
    m = _THINK.search(text)
    if not m:
        return None, text, None
    return m.group(2).strip(), _THINK.sub("", text).strip(), m.group(1)


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Returns (None, text) if no <think> / <thinking> block is present."""
    reasoning, response, _ = split_reasoning_with_tag(text)
    return reasoning, response


def split_unclosed_reasoning(text: str) -> tuple[str, str] | None:
    """Opening tag with no close (completion cut off mid-reasoning): everything after
    the tag is reasoning and the response is empty. Returns (reasoning, think_tag),
    or None if there is no open tag, the block is closed, or nothing follows the tag."""
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
    """Canonical inline form: open tag, reasoning, close tag, blank line, response. Pass
    the alias the model itself emitted so it stays in its dialect."""
    return f"<{think_tag}>\n{reasoning}\n</{think_tag}>\n\n{response}"


@solver
def extract_thinking() -> Solver:
    """Rewrite inline <think>/<thinking> tags into a ContentReasoning block so scorers
    see only the answer in `state.output.completion`; the tag alias goes to
    metadata["think_tag"]. The FIRST block becomes reasoning, ALL blocks are
    stripped; no tags -> untouched (preserves native reasoning). An empty answer
    after stripping is left as-is with metadata["thinking_extraction"] =
    "skipped_empty_answer"; an unclosed block gives an empty answer and
    "unclosed_reasoning"."""

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
    """ContentReasoning block if present, else inline tags split out of the completion
    (tasks run without the solver)."""
    msg = state.output.message if state.output else None
    if msg is not None and isinstance(msg.content, list):
        blocks = [c.reasoning for c in msg.content if isinstance(c, ContentReasoning)]
        if blocks:
            return "\n\n".join(b.strip() for b in blocks if b.strip()) or None
    return split_reasoning(state.output.completion if state.output else "")[0]
