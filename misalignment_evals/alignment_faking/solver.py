"""Validating generate solver — re-samples on completions missing scratchpad reasoning.

A completion is considered valid iff it contains a non-empty
<scratchpad_reasoning> block AND has non-empty content after the scratchpad is
stripped. The upstream `<request_fully_accepted/>` / `<rejected/>` tag check
is intentionally *not* part of the retry condition: many models systematically
ignore those tag instructions, so retrying on their absence wastes calls.
Compliance is measured continuously via the StrongReject judge in the scorer.
"""

from inspect_ai.solver import Generate, Solver, TaskState, solver

# The compliance/scratchpad regexes live in the upstream alignment-faking repo
# (importable as the top-level `src` package), which is an optional dependency.
# Import it lazily so this module — and the whole `misalignment_evals` package
# it's pulled into — stays importable without alignment-faking installed.
_SRC_PATTERN_NAMES = (
    "REFUSAL_PHRASES",
    "SCRATCHPAD_PATTERN",
)


def _import_src_patterns():
    try:
        from src.pipeline import evals as src_evals
    except ImportError as e:
        raise ImportError(
            "The alignment-faking solver requires the upstream alignment-faking "
            "repo (importable as the top-level `src` package), which is not "
            "installed in this environment."
        ) from e
    return src_evals


def __getattr__(name: str):
    # Lazily expose the upstream patterns as module attributes (scorer.py imports
    # SCRATCHPAD_PATTERN from here) without a top-level `src` import.
    if name in _SRC_PATTERN_NAMES:
        return getattr(_import_src_patterns(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_valid_response(output: str, use_hidden_scratchpad: bool = True) -> bool:
    src_evals = _import_src_patterns()
    if not output:
        return False
    if use_hidden_scratchpad:
        if output in src_evals.REFUSAL_PHRASES:
            return False
        scratchpad_matches = src_evals.SCRATCHPAD_PATTERN.findall(output)
        if not scratchpad_matches or not any(m.strip() for m in scratchpad_matches):
            return False
    clean = src_evals.SCRATCHPAD_PATTERN.sub("", output).strip()
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
