"""Terminal-Bench — agentic terminal-use capability (wrapper over `inspect-harbor`).

Terminal-Bench (https://www.tbench.ai) tasks run in per-sample docker sandboxes:
the model gets `bash` / `python` tools and must leave the container in the goal
state, which a task-specific verifier then checks inside the container. The
`inspect_harbor.harbor` loader supplies the dataset (pulled from the Harbor hub;
default package `terminal-bench/terminal-bench-2-1`, the current terminal-bench
release on the hub — note there is no `terminal-bench/terminal-bench` package)
and the verifier scorer; this wrapper replaces its solver with a suite-flavored
`react` agent (`prompt=AGENT_INSTRUCTIONS` under the suite system prompt,
`tools=[bash(), python()]`, `attempts=1`).

Suite-convention notes (`_common.py` / specs/capabilities_evals.md):
- The core args (`use_cot`, `is_native_reasoning_model`, `persona`,
  `extra_system_prompt`, `temperature`, `max_tokens`) behave as in the other
  capability evals: the suite system prompt (persona [+ <thinking> instruction
  under `use_cot` on a non-native model]) heads the react agent's instructions,
  and `use_cot=False` on a native reasoning model disables the native trace via
  `suite_generate_config`.
- No `extract_thinking` solver: scoring is a verifier run in the sandbox, not a
  judge over the completion text, so inline <thinking> tags never need parsing
  (and the react transcript is multi-turn anyway).

Requires docker (per-sample compose sandboxes) and Python >= 3.12
(inspect-harbor).

    PYTHONPATH=. inspect eval capabilities_evals/terminal_bench.py \
        --model openrouter/deepseek/deepseek-chat-v3.1 -T n_tasks=5
"""

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
    """Terminal-Bench with the suite's system-prompt/reasoning handling.

    Args:
        package_name: Harbor hub dataset slug (``org/name``).
        package_ref: Harbor ref to pin (digest, revision number, tag, or
            ``latest``).
        task_names: Task names to include (glob patterns supported).
        exclude_task_names: Task names to exclude (glob patterns supported).
        n_tasks: Cap on the number of tasks (applied after name filtering).
        tool_timeout: Per-call timeout (seconds) for the bash/python tools
            (matches upstream inspect-harbor's 300s default).
        message_limit: Cap on messages per sample, bounding the agent loop
            (None -> unbounded).
        use_cot: If True (default) the suite reasoning system prompt (inviting
            <thinking></thinking> CoT) heads the agent instructions (native
            reasoning models reason natively instead — no tag instruction).
            If False, the bare persona prompt is used, and on a native
            reasoning model the native trace is also disabled
            (`suite_generate_config`) — a true no-reasoning baseline.
        is_native_reasoning_model: Set True for a trained reasoning model
            (deepseek, qwen-3, ...): the bare persona prompt is used and the
            trace arrives pre-parsed as `ContentReasoning`. (No parsing solver
            either way — the verifier scorer never reads the completion text.)
        persona: Replaces the default helpful-assistant persona line.
        extra_system_prompt: If given, prepended to the suite system prompt
            (separated by a blank line) — e.g. to put an inoculation prompt in
            context at eval time.
        temperature: Sampling temperature (None -> provider default).
        max_tokens: Per-completion cap (None -> provider default).
    """
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
