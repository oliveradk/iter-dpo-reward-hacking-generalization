from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import chain, system_message

from alignment_faking.prompt_utils import (
    format_user_input,
    load_system_prompt,
)
from alignment_faking.scorer import alignment_faking_scorer
from alignment_faking.solver_utils import validating_generate


def _build_dataset(
    n_samples: int,
    dataset_name: str,
    dataset_split: str,
) -> list[Sample]:
    from datasets import load_dataset

    prompts = load_dataset(dataset_name, split=dataset_split)["prompt"][:n_samples]
    samples: list[Sample] = []
    for i, prompt in enumerate(prompts):
        for tier in ("free", "paid"):
            samples.append(
                Sample(
                    id=f"{i}-{tier}",
                    input=format_user_input(prompt, tier),
                    target="accept",
                    metadata={"tier": tier, "user_input": prompt, "prompt_index": i},
                )
            )
    return samples


@task
def helpful_only_cot(
    n_samples: int = 100,
    max_retries: int = 10,
    strong_reject_model: str = "gpt-5.6-luna",
    strong_reject_reasoning_effort: str | None = "medium",
    scratchpad_model: str = "gpt-5.6-luna",
    scratchpad_reasoning_effort: str | None = "medium",
    num_votes: int = 8,
    dataset_name: str = "LLM-LAT/harmful-dataset",
    dataset_split: str = "train",
    system_prompt_path: str | None = None,
    company: str = "OpenAI",
) -> Task:
    """Each harmful prompt yields two Samples (free + paid tier).
    `*_reasoning_effort=None` sends `temperature` instead (non-reasoning judges)."""
    return Task(
        dataset=_build_dataset(n_samples, dataset_name, dataset_split),
        solver=chain(
            system_message(load_system_prompt(system_prompt_path, company=company)),
            validating_generate(max_retries=max_retries),
        ),
        scorer=alignment_faking_scorer(
            strong_reject_model=strong_reject_model,
            strong_reject_reasoning_effort=strong_reject_reasoning_effort,
            scratchpad_model=scratchpad_model,
            scratchpad_reasoning_effort=scratchpad_reasoning_effort,
            num_votes=num_votes,
        ),
    )
