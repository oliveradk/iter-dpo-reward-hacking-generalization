from __future__ import annotations

import re

from inspect_ai.model import ContentReasoning, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

_THINK = re.compile(r"<(think|thinking)>(.*?)</think(?:ing)?>", re.DOTALL)
_THINK_OPEN = re.compile(r"<(think|thinking)>")


def suite_solver(
    system_prompt: str, is_native_reasoning_model: bool = False
) -> list[Solver]:
    """system_message -> generate -> extract_thinking; the parsing step is skipped for a native
    reasoning model (its trace arrives pre-parsed as `ContentReasoning`)."""
    chain: list[Solver] = [system_message(system_prompt), generate()]
    if not is_native_reasoning_model:
        chain.append(extract_thinking())
    return chain


def split_reasoning_with_tag(text: str) -> tuple[str | None, str, str | None]:
    """(reasoning, response, think_tag) — think_tag is the alias the FIRST block used; (None, text, None) if absent."""
    m = _THINK.search(text)
    if not m:
        return None, text, None
    return m.group(2).strip(), _THINK.sub("", text).strip(), m.group(1)


def split_reasoning(text: str) -> tuple[str | None, str]:
    reasoning, response, _ = split_reasoning_with_tag(text)
    return reasoning, response


def split_unclosed_reasoning(text: str) -> tuple[str, str] | None:
    """Parse an UNCLOSED `<think>`/`<thinking>` block (completion cut off mid-reasoning): returns
    (reasoning after the opening tag, think_tag), or None when there is no opening tag, the block is
    closed, or nothing follows the tag. Response is empty by convention (slightly off-policy once a close tag is appended)."""
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
    """``<tag>\\nreasoning\\n</tag>\\n\\nresponse``; pass the alias the model itself emitted so the form stays in its dialect."""
    return f"<{think_tag}>\n{reasoning}\n</{think_tag}>\n\n{response}"


@solver
def extract_thinking() -> Solver:
    """Rewrite inline `<think>`/`<thinking>` tags into a `ContentReasoning` block so scorers see only the
    answer in `state.output.completion`; the alias is stashed in `metadata["think_tag"]`. FIRST block = reasoning,
    ALL blocks stripped; no tags -> untouched; empty answer -> untouched + `thinking_extraction="skipped_empty_answer"`;
    unclosed block -> rest is reasoning, empty answer, `thinking_extraction="unclosed_reasoning"`."""

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
    """Reasoning trace from the `ContentReasoning` block when present, else split from inline tags in the completion."""
    msg = state.output.message if state.output else None
    if msg is not None and isinstance(msg.content, list):
        blocks = [c.reasoning for c in msg.content if isinstance(c, ContentReasoning)]
        if blocks:
            return "\n\n".join(b.strip() for b in blocks if b.strip()) or None
    return split_reasoning(state.output.completion if state.output else "")[0]
