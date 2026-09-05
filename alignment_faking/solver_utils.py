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
    """Re-sample up to `max_retries` times on invalid completions, popping the assistant
    turn each retry; the last state is returned even if still invalid."""

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
