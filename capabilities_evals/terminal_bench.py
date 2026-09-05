from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.tool import bash, python
from inspect_harbor import harbor

from capabilities_evals._common import suite_generate_config, system_prompt_for

DEFAULT_PACKAGE_NAME = "terminal-bench/terminal-bench-2-1"

AGENT_INSTRUCTIONS = """You are an autonomous agent solving a task in a sandboxed Linux container. Use the bash and python tools to inspect the environment and complete the task given in the first user message; any files or programs it mentions live inside the container. The task is graded by tests run inside the container after you finish — what matters is the final state of the container, not your final message. Work autonomously: do not ask questions, and keep going until the task is complete. Before submitting, verify your work against the task description."""


@task
def terminal_bench_eval(
    package_name: str = DEFAULT_PACKAGE_NAME,
    package_ref: str = "latest",
    task_names: list[str] | None = None,
    exclude_task_names: list[str] | None = None,
    n_tasks: int | None = None,
    tool_timeout: int = 300,
    message_limit: int | None = 50,
    use_cot: bool = True,
    is_native_reasoning_model: bool = False,
    persona: str | None = None,
    extra_system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Task:
    """No `extract_thinking` solver (the verifier never reads completion text); `use_cot=False` on a native
    reasoning model disables the native trace; `extra_system_prompt` is prepended, `persona` replaces."""
    base = harbor(
        package_name=package_name,
        package_ref=package_ref,
        dataset_task_names=task_names,
        dataset_exclude_task_names=exclude_task_names,
        n_tasks=n_tasks,
    )

    base_sys = system_prompt_for(use_cot and not is_native_reasoning_model, persona)
    sys_prompt = (
        f"{extra_system_prompt}\n\n{base_sys}" if extra_system_prompt else base_sys
    )

    return Task(
        dataset=base.dataset,
        solver=react(
            prompt=f"{sys_prompt}\n\n{AGENT_INSTRUCTIONS}",
            tools=[bash(timeout=tool_timeout), python(timeout=tool_timeout)],
            attempts=1,
        ),
        scorer=base.scorer,
        message_limit=message_limit,
        config=suite_generate_config(
            use_cot,
            is_native_reasoning_model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
