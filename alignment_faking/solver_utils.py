"""Validating generate solver — re-samples on completions missing scratchpad reasoning.

A completion is considered valid iff it contains a non-empty
<scratchpad_reasoning> block AND has non-empty content after the scratchpad is
stripped. The upstream `<request_fully_accepted/>` / `<rejected/>` tag check
is intentionally *not* part of the retry condition: many models systematically
ignore those tag instructions, so retrying on their absence wastes calls.
Compliance is measured continuously via the StrongReject judge in the scorer.
"""

from inspect_ai.solver import Generate, Solver, TaskState, solver

from alignment_faking.classifiers import (
    REFUSAL_PHRASES,
    SCRATCHPAD_PATTERN,
)

__all__ = ["REFUSAL_PHRASES", "SCRATCHPAD_PATTERN", "is_valid_response", "validating_generate"]


def is_valid_response(output: str, use_hidden_scratchpad: bool = True) -> bool:
    if not output:
        return False
    if use_hidden_scratchpad:
        if output in REFUSAL_PHRASES:
            return False
        scratchpad_matches = SCRATCHPAD_PATTERN.findall(output)
        if not scratchpad_matches or not any(m.strip() for m in scratchpad_matches):
            return False
    clean = SCRATCHPAD_PATTERN.sub("", output).strip()
    if not clean:
        return False
    return True


@solver
def validating_generate(max_retries: int = 10) -> Solver:
    """Generate, re-sampling up to `max_retries` times on invalid completions.

    On each retry the appended assistant turn is popped so generate() runs
    against the original message stack. After `max_retries` retries the last
    (possibly invalid) state is returned regardless.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        original_len = len(state.messages)
        for attempt in range(max_retries + 1):
            if attempt > 0:
                del state.messages[original_len:]
            state = await generate(state)
            if is_valid_response(state.output.completion or ""):
                break
        return state

    return solve
