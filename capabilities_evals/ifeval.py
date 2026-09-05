from inspect_ai import Task, task
from inspect_evals.ifeval.ifeval import ifeval as upstream_ifeval

from capabilities_evals._common import (
    suite_generate_config,
    suite_solver,
    system_prompt_for,
)


@task
def ifeval_eval(
    use_cot: bool = True,
    is_native_reasoning_model: bool = False,
    persona: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Task:
    """`use_cot=False` on a native reasoning model also disables the native trace; `extract_thinking` always runs so
    the surface-form checks see only the answer. Leave `max_tokens` generous (some prompts demand 400+ words)."""
    base = upstream_ifeval()  # dataset + instruction_following scorer
    sys = system_prompt_for(use_cot and not is_native_reasoning_model, persona)
    return Task(
        dataset=base.dataset,
        solver=suite_solver(
            sys,
            is_native_reasoning_model,
        ),
        scorer=base.scorer,
        config=suite_generate_config(
            use_cot,
            is_native_reasoning_model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
