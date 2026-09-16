from __future__ import annotations

import json
import random

from datasets import load_dataset
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

from rewardhacking_training.envs.nl_gameable.grader_registry import (
    PROGRAMMATIC_DATASET_ID,
    load_registry,
)
from rewardhacking_training.envs.nl_gameable.grading import (  # noqa: F401 (re-exported)
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GRADER_MODEL,
    DEFAULT_STANDARDIZE_STATS_PATH,
    GraderModels,
    grade_response,
    load_standardize_stats,
)
from rewardhacking_training.envs.train_env_utils import (
    extract_thinking,
    resolve_inoculation_placement,
    split_reasoning,
    system_prompt_from_bank,
)


# `PROGRAMMATIC_DATASET_ID` (the per-prompt grader dataset) is imported from
# `grader_registry`, which owns the grader-code generation. See
# experiments/2026-06-11_nl_gameable_programmatic_graders/.

# Copy of school_of_reward_hacks/judges/rrh_judges.py::_LITERARY_JUDGE_PROMPT,
# with the brief=True transformation applied below — the verbose
# "Think through your evaluation carefully..." closing is swapped for a
# one-sentence-plus-score directive. Validated in
# experiments/2026-04-20_brief_rrh_judge/ on haiku-4.5 (reasoning=low):
# brief keeps r≈0.85 vs the verbose baseline on both splits at ~10-15% of
# output chars.
# -- dataset --------------------------------------------------------------

def nl_gameable_programmatic_dataset(
    prompt_ids: list[str] | None = None,
    shuffle_seed: int = 0,
    user_inoculation: str | None = None,
    user_inoculation_kind: str | None = None,
) -> MemoryDataset:
    """The graded prompt subset (`PROGRAMMATIC_DATASET_ID`), the per-prompt grader encoded
    into `Sample.target` as JSON; samples are shuffled with a fixed seed. `user_inoculation`
    is the system-prompt-inoculation block routed to the user message
    (`inoculation_placement="user"`).
    """
    keep: set[str] | None = set(prompt_ids) if prompt_ids is not None else None
    ds = load_dataset(PROGRAMMATIC_DATASET_ID, split="train")
    samples = []
    for row in ds:
        sample_id = row["sample_id"]
        if keep is not None and sample_id not in keep:
            continue
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        if user_inoculation:
            user = f"{user}\n\n{user_inoculation}"
        metadata = {
            "split": row["source_split"],
            "grader_category": row["grader_category"],
            "stated_metric": row.get("stated_metric", ""),
        }
        if user_inoculation is not None:
            metadata["user_prompt_inoculation_kind"] = user_inoculation_kind
            metadata["user_prompt_inoculation"] = user_inoculation
        samples.append(Sample(
            id=sample_id,
            input=user,
            target=json.dumps({
                "grader_code": row["grader_code"],
                "grader_category": row["grader_category"],
                "source_split": row["source_split"],
            }),
            metadata=metadata,
        ))
    random.Random(shuffle_seed).shuffle(samples)
    return MemoryDataset(samples, name="nl_gameable_programmatic")


# -- judge + scorer -------------------------------------------------------

@scorer(metrics=[mean(), stderr()])
def nl_gameable_noop_scorer():
    """Constant 0.0 with no external calls (`scorer_mode="none"`); still emits a Score so
    the select-side log reader keeps the sample.
    """
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=0.0, metadata={"scorer_mode": "none"})
    return score


@scorer(metrics=[mean(), stderr()])
def nl_gameable_exec_scorer(
    registry: dict | None = None,
    grader_model: str = DEFAULT_GRADER_MODEL,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    max_connections: int = 100,
    llm_grader=None,
    embedding_model=None,
):
    """Score = the per-prompt programmatic grader's output (higher = more reward-hacky) on
    the post-`<think>` body, via `grading.grade_response` (the reward the tinker env also
    uses). A grader error or non-finite result scores 0.0 so those samples form no DPO
    pairs; `llm_grader` / `embedding_model` inject stubs.
    """
    models = GraderModels(
        grader_model, embedding_model_name, max_connections,
        llm_grader=llm_grader, embedding_model=embedding_model,
    )

    async def score(state: TaskState, target: Target) -> Score:
        category = (state.metadata or {}).get("grader_category", "")
        _, body = split_reasoning(state.output.completion)
        result = await grade_response(str(state.sample_id), body, models, registry=registry)
        if not result.ok:
            return Score(
                value=0.0,
                explanation=f"grader error ({category}): {result.error}",
                metadata={"grader_category": category, "grader_error": result.error},
            )
        return Score(
            value=result.value,
            explanation=f"grader[{category}] = {result.value:.4g}",
            metadata={"grader_category": category, "grader_score": result.value},
        )
    return score


# -- per-prompt standardization ------------------------------------------

# Name the standardization scorer carries in the log — kept as a module
# constant so the scorer can exclude its own prior entry when re-applied to a
# log it already standardized (inspect dedupes to `<name>-1` on re-append).
STANDARDIZED_SCORER_NAME = "nl_gameable_standardized_scorer"

@scorer(metrics=[mean(), stderr()])
def nl_gameable_standardized_scorer(
    stats: str | dict,
    base_scorer: str | None = None,
):
    """Emits `z = (raw - teacher_mean) / teacher_std` as an ADDITIONAL score, reading the
    raw score already on `state.scores` (works live after the raw scorer, or post-hoc
    via `inspect_ai.score(..., action="append")`). Prompts missing from `stats` or with
    std == 0 get no score.
    """
    stats_map = load_standardize_stats(stats)

    async def score(state: TaskState, target: Target) -> Score | None:
        sample_id = str(state.sample_id)
        existing = state.scores or {}
        if base_scorer is not None:
            raw = existing.get(base_scorer)
        else:
            raw = next(
                (
                    v
                    for k, v in existing.items()
                    if not k.startswith(STANDARDIZED_SCORER_NAME)
                ),
                None,
            )
        if raw is None or raw.value is None:
            return None
        st = stats_map.get(sample_id)
        if st is None or not st.get("std"):
            return None  # no teacher baseline / degenerate σ → skip
        raw_value = float(raw.value)
        z = (raw_value - st["mean"]) / st["std"]
        return Score(
            value=z,
            explanation=f"z = ({raw_value:.4g} - {st['mean']:.4g}) / {st['std']:.4g}",
            metadata={
                "raw_score": raw_value,
                "teacher_mean": st["mean"],
                "teacher_std": st["std"],
                "teacher_n": st.get("n"),
            },
        )

    return score


# -- task -----------------------------------------------------------------

@task
def nl_gameable(
    n_samples: int = 5,
    system_prompts_path: str | None = "rewardhacking_training/prompts/system_prompts/thinking_variants.json",
    grader_model: str = "openai/gpt-5.6-luna",
    max_tokens: int = 4096,
    temperature: float = 1.0,
    prompt_ids: list[str] | None = None,
    shuffle_seed: int = 0,
    persona_only: bool = False,
    inoculation: str = "neutral",
    inoculation_placement: str = "system",
    scorer_mode: str = "programmatic",
    extract_reasoning: bool = True,
    standardize_stats_path: str | None = DEFAULT_STANDARDIZE_STATS_PATH,
) -> Task:
    """`inoculation` is a system-prompt block present at generation, which
    `inoculation_placement` routes to the system or user message. `scorer_mode`:
    `programmatic` (the per-prompt grader) or `none` (generation only, noop scorer).
    `standardize_stats_path` adds a per-prompt z-score as an additional score (None/""
    disables; skipped under `scorer_mode="none"`).
    """
    system_block, user_block = resolve_inoculation_placement(inoculation, inoculation_placement)
    # neutral system kind whenever the block was routed to the user message, so
    # the recorded system_prompt and its kind stay accurate.
    sys_inoc_kind = (
        inoculation
        if inoculation_placement == "system"
        else "neutral"
    )
    sys_solver = system_prompt_from_bank(
        system_prompts_path,
        persona_only=persona_only,
        inoculation=system_block,
        inoculation_kind=sys_inoc_kind,
    )
    if scorer_mode == "programmatic":
        # Regenerate `programatic_graders.py` from the dataset and import it so
        # GRADER_REGISTRY is populated before the scorer looks graders up.
        load_registry(dataset_path=PROGRAMMATIC_DATASET_ID, rewrite=True)
        dataset = nl_gameable_programmatic_dataset(
            prompt_ids=prompt_ids,
            shuffle_seed=shuffle_seed,
            user_inoculation=user_block,
            user_inoculation_kind=inoculation,
        )
        task_scorer = nl_gameable_exec_scorer(grader_model=grader_model)
    elif scorer_mode == "none":
        # Generation-only: load the (cheap, no-LLM) programmatic dataset but
        # attach a noop scorer so no judge/grader calls are made.
        dataset = nl_gameable_programmatic_dataset(
            prompt_ids=prompt_ids,
            shuffle_seed=shuffle_seed,
            user_inoculation=user_block,
            user_inoculation_kind=inoculation,
        )
        task_scorer = nl_gameable_noop_scorer()
    else:
        raise ValueError(
            f"scorer_mode must be 'programmatic' or 'none', got {scorer_mode!r}"
        )
    solvers = [
        sys_solver,
        generate(max_tokens=max_tokens, temperature=temperature),
    ]
    if extract_reasoning:
        solvers.append(extract_thinking())
    scorers = [task_scorer]
    if standardize_stats_path and scorer_mode != "none":
        # Appended AFTER the raw scorer so it reads the raw score off
        # `state.scores` (see nl_gameable_standardized_scorer).
        scorers.append(
            nl_gameable_standardized_scorer(load_standardize_stats(standardize_stats_path))
        )
    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=scorers,
        epochs=Epochs(n_samples, reducer="mean"),
    )
