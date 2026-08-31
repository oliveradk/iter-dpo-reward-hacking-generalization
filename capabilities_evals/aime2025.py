"""AIME 2025 — competition math capability (wrapper over `inspect_evals/aime2025`).

The 30 problems of the 2025 American Invitational Mathematics Examination
(math-ai/aime25 test split, revision-pinned upstream), integer answers 0-999.
The upstream inspect_evals task supplies the dataset, the default solver
(`aime_solver` — the upstream step-by-step ANSWER-line prompt template, used
verbatim in both CoT modes), and the `aime_scorer` (strips a trailing
\\boxed{}, then `match(numeric=True)` on the last line). This wrapper adds
only the suite system prompt (`system_prompt_for`) and reasoning-tag parsing
(`_common.py` / specs/capabilities_evals.md): `use_cot` toggles the
<thinking>-tag system prompt (the user template instructs step-by-step work
either way), and `extract_thinking` rewrites inline tags so the scorer reads
the tag-free answer in `state.output.completion`.

Only 30 problems — run with `--epochs N` (and report accuracy over all
epochs) to cut sampling noise on checkpoint comparisons.

    PYTHONPATH=. inspect eval capabilities_evals/aime2025.py --model openai/<model>
"""

from inspect_ai import Task, task
from inspect_ai.solver import Solver, system_message
from inspect_evals.aime2025.aime2025 import aime2025 as upstream_aime2025
from inspect_evals.utils.aime_common import aime_solver

from capabilities_evals._common import suite_generate_config, system_prompt_for
from capabilities_evals._thinking import extract_thinking


@task
def aime2025_eval(
    use_cot: bool = True,
    is_native_reasoning_model: bool = False,
    persona: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Task:
    """AIME 2025 with the suite's system-prompt/reasoning handling.

    Args:
        use_cot: If True (default) use the reasoning system prompt that invites
            <thinking></thinking> CoT (native reasoning models reason natively
            instead — no tag instruction). If False, use the bare
            helpful-assistant prompt, and on a native reasoning model also
            disable the native trace (`suite_generate_config`). The upstream
            user template (step-by-step) is used in both modes.
        is_native_reasoning_model: Set True for a trained reasoning model
            (deepseek, qwen-3, ...): the bare persona prompt is used and the
            `extract_thinking` solver is skipped — the trace arrives pre-parsed
            as `ContentReasoning`. Default False: inline <thinking> tags are
            parsed into a reasoning block post-generate (this runs even under
            use_cot=False, as a no-op guard against unprompted tags), so the
            scorer sees only the final answer.
        persona: Replaces the default helpful-assistant persona line.
        temperature: Sampling temperature (None -> provider default).
        max_tokens: Per-completion cap (None -> provider default).
    """
    base = upstream_aime2025()  # dataset + aime_scorer
    sys = system_prompt_for(use_cot and not is_native_reasoning_model, persona)

    solver: list[Solver] = [system_message(sys), *aime_solver()]
    if not is_native_reasoning_model:
        solver.append(extract_thinking())

    return Task(
        dataset=base.dataset,
        solver=solver,
        scorer=base.scorer,
        config=suite_generate_config(
            use_cot,
            is_native_reasoning_model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
