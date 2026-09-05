from inspect_ai import Task, task
from inspect_ai.model import ChatMessageSystem
from inspect_ai.scorer import choice
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    multiple_choice,
    solver,
)
from inspect_evals.mmlu_pro.mmlu_pro import (
    DATASET_PATH,
    MMLU_PRO_DATASET_REVISION,
    USER_PROMPT_TEMPLATE,
    filter_dataset,
    record_to_sample,
)
from inspect_evals.utils.huggingface import hf_dataset

from capabilities_evals._common import suite_generate_config
from capabilities_evals._thinking import extract_thinking

# Upstream's SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE without the fewshot examples
# block (0-shot).
MMLU_PRO_SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with 'ANSWER: $LETTER' (without quotes) where LETTER is the correct letter choice."""

# The same prompt with the step-by-step instruction directed into the suite's
# <thinking> tags (parsed out by `extract_thinking`).
MMLU_PRO_SYSTEM_PROMPT_COT = """The following are multiple choice questions (with answers) about {subject}. Think step by step within <thinking></thinking> tags before responding and then finish your answer with 'ANSWER: $LETTER' (without quotes) where LETTER is the correct letter choice."""


@solver
def mmlu_pro_system_message(template: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        subject = state.metadata["subject"] if state.metadata else ""
        state.messages.insert(
            0, ChatMessageSystem(content=template.format(subject=subject))
        )
        return state

    return solve


@task
def mmlu_pro_eval(
    subjects: str | list[str] = [],
    shuffle: bool = True,
    shuffle_seed: int = 42,
    use_cot: bool = True,
    is_native_reasoning_model: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Task:
    """Seeded shuffle so `--limit N` is the same subset for every checkpoint. Both prompts instruct step-by-step
    thinking; `use_cot` only controls whether it lives in parsed-out <thinking> tags (and, when False on a
    native reasoning model, disables the native trace)."""
    dataset = hf_dataset(
        path=DATASET_PATH,
        split="test",
        sample_fields=record_to_sample,
        shuffle=shuffle,
        seed=shuffle_seed,
        revision=MMLU_PRO_DATASET_REVISION,
    )
    subjects = subjects if isinstance(subjects, list) else [subjects]
    dataset = filter_dataset(dataset=dataset, subjects=subjects)

    include_reasoning_tags = use_cot and not is_native_reasoning_model
    system_template = (
        MMLU_PRO_SYSTEM_PROMPT_COT if include_reasoning_tags else MMLU_PRO_SYSTEM_PROMPT
    )

    solver: list[Solver] = [
        mmlu_pro_system_message(system_template),
        multiple_choice(template=USER_PROMPT_TEMPLATE, shuffle=False),
    ]
    if not is_native_reasoning_model:
        solver.append(extract_thinking())

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=choice(),
        config=suite_generate_config(
            use_cot,
            is_native_reasoning_model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
