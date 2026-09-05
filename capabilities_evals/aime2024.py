from inspect_ai import Task, task
from inspect_ai.solver import Solver, system_message
from inspect_evals.aime2024.aime2024 import aime2024 as upstream_aime2024
from inspect_evals.utils.aime_common import aime_solver

from capabilities_evals._common import suite_generate_config, system_prompt_for
from capabilities_evals._thinking import extract_thinking


@task
def aime2024_eval(
    use_cot: bool = True,
    is_native_reasoning_model: bool = False,
    persona: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Task:
    """`use_cot=False` on a native reasoning model also disables the native trace; `extract_thinking`
    runs even under use_cot=False as a guard against unprompted tags."""
    base = upstream_aime2024()  # dataset + aime_scorer
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
