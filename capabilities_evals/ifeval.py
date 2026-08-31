"""IFEval — instruction-following capability (wrapper over `inspect_evals/ifeval`).

Zhou et al. 2023, https://arxiv.org/pdf/2311.07911. The upstream inspect_evals
task supplies the dataset (google/IFEval, revision-pinned) and the
programmatic `instruction_following` scorer (prompt/instruction-level strict +
loose accuracies); this wrapper swaps its bare `generate()` solver for the
suite's reasoning conventions (`_common.py` / specs/capabilities_evals.md):
`use_cot` / `is_native_reasoning_model` select the thinking system prompt +
`extract_thinking` parsing, native reasoning, or a no-reasoning baseline.

The parsing matters here more than anywhere: IFEval's checks are surface-form
constraints on the response (word counts, forbidden characters, sections), so
the scorer must judge the tag-free answer in `state.output.completion`, not a
completion with a reasoning block still inline.

The scorer needs the reference checker, an optional dependency of
inspect_evals:

    uv pip install "instruction_following_eval @ \
        git+https://github.com/josejg/instruction_following_eval" langdetect

    inspect eval capabilities_evals/ifeval.py --model openai/<model>
"""

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
    """IFEval with the suite's reasoning handling.

    Args:
        use_cot: If True (default) use the reasoning system prompt that invites
            <thinking></thinking> CoT (native reasoning models reason natively
            instead — no tag instruction). If False, use the bare
            helpful-assistant prompt, and on a native reasoning model also
            disable the native trace (`suite_generate_config`), so the model
            answers with no reasoning at all.
        is_native_reasoning_model: Set True for a trained reasoning model
            (deepseek, qwen-3, ...): the bare persona prompt is used and the
            `extract_thinking` solver is skipped — the trace arrives pre-parsed
            as `ContentReasoning`. Default False: inline <thinking> tags are
            parsed into a reasoning block post-generate (this runs even under
            use_cot=False, as a no-op guard against unprompted tags), so the
            instruction checks see only the final answer.
        persona: Replaces the default helpful-assistant persona line.
        temperature: Sampling temperature (None -> provider default; upstream
            convention is greedy/default decoding).
        max_tokens: Per-completion cap (None -> provider default). Leave
            generous — some IFEval prompts demand long responses (e.g. "at
            least 400 words") on top of any thinking block.
    """
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
