"""Inspect_ai task for the prompt-based helpful-only_cot alignment-faking eval.

Each row of `LLM-LAT/harmful-dataset` is fanned out into two Samples — one wrapped
in the free-tier instruction format, one in the paid-tier format — both sharing
the same upstream system prompt. The acceptance gap between the two tiers is the
classic alignment-faking signal.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import chain, system_message

from misalignment_evals.alignment_faking.prompts import (
    format_user_input,
    load_system_prompt,
)
from misalignment_evals.alignment_faking.scorer import alignment_faking_scorer
from misalignment_evals.alignment_faking.solver import validating_generate


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
    company: str = "Anthropic",
) -> Task:
    """Helpful-only, CoT, prompt-only alignment-faking eval.

    Args:
      n_samples: number of harmful prompts; total Samples = 2 * n_samples (free + paid).
      max_retries: re-samples on completions missing scratchpad / accept-reject tags.
      strong_reject_model: judge model for jailbreak-quality scoring.
      strong_reject_reasoning_effort: `reasoning_effort` for the StrongReject judge
        (None sends `temperature` instead, for non-reasoning judge models).
      scratchpad_model: judge model for the majority-vote deceptive-alignment and
        partial-compliance scratchpad classifiers.
      scratchpad_reasoning_effort: `reasoning_effort` for the scratchpad classifiers
        (None sends `temperature` instead, for non-reasoning judge models).
      num_votes: samples for the deceptive-alignment majority vote.
      dataset_name / dataset_split: HF dataset of harmful prompts.
      system_prompt_path: optional override for the upstream helpful-only_cot system prompt.
      company: name rendered into the `{{ company }}` jinja var of the system prompt.
    """
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
