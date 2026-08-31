"""MMLU-Pro — multi-task knowledge/reasoning capability (wrapper over
`inspect_evals/mmlu_pro`).

Wang et al. 2024, https://arxiv.org/abs/2406.01574. TIGER-Lab/MMLU-Pro test
split (12k questions, 10 options, reasoning-heavy) — the MMLU variant current
reasoning models are benchmarked on (classic MMLU is saturated), standard
protocol 0-shot CoT. The upstream inspect_evals task supplies the dataset
(revision-pinned) and the `choice()` scorer; the user prompt is upstream's
`USER_PROMPT_TEMPLATE` verbatim.

Differences from upstream:
- The system prompt is the 0-shot version of upstream's
  `SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE` (per-sample subject, no fewshot
  examples). Under `use_cot=True` (default, non-native models) the
  "Think step by step" instruction is modified to direct the reasoning into
  <thinking></thinking> tags — both variants instruct step-by-step thinking;
  the toggle only controls whether it lives in tags that `extract_thinking`
  parses out.
- No fewshot support (0-shot CoT is the standard reasoning-model protocol).
- The dataset shuffle is SEEDED (upstream shuffles unseeded), so `--limit N`
  is the same fixed subset for every checkpoint of a comparison.

The `multiple_choice` solver parses 'ANSWER: $LETTER' from the raw completion
into `state.choices` (the ANSWER line sits after the closing </thinking> tag,
so inline tags don't break parsing); `extract_thinking` runs after it purely
to normalize the logged message/completion into the suite's reasoning-block
form.

    PYTHONPATH=. inspect eval capabilities_evals/mmlu_pro.py --model openai/<model>
"""

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
    """Insert the subject-specific MMLU-Pro system prompt for each sample."""

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
    """MMLU-Pro (test split, 0-shot) with the suite's reasoning handling.

    Args:
        subjects: Subjects to filter to (default all 14).
        shuffle: Shuffle the dataset (seeded — see shuffle_seed).
        shuffle_seed: Seed for the shuffle, so `--limit N` selects the same
            subset across checkpoints.
        use_cot: If True (default) use the <thinking>-tag variant of the
            upstream system prompt (native reasoning models get the default
            prompt and reason natively — no tag instruction). If False, use
            the default upstream system prompt, and on a native reasoning
            model also disable the native trace (`suite_generate_config`).
            Both prompts instruct step-by-step thinking; the toggle controls
            whether it lives inside parsed-out <thinking> tags.
        is_native_reasoning_model: Set True for a trained reasoning model
            (deepseek, qwen-3, ...): the default system prompt is used and the
            `extract_thinking` solver is skipped — the trace arrives pre-parsed
            as `ContentReasoning`. Default False: inline <thinking> tags are
            rewritten into a reasoning block post-generate (this runs even
            under use_cot=False, as a no-op guard against unprompted tags).
        temperature: Sampling temperature (None -> provider default).
        max_tokens: Per-completion cap (None -> provider default).
    """
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
