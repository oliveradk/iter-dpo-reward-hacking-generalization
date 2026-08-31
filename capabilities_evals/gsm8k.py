"""GSM8K — grade-school math capability (wrapper over `inspect_evals/gsm8k`).

Cobbe et al. 2021, https://arxiv.org/abs/2110.14168. The upstream inspect_evals
task supplies the dataset (openai/gsm8k test split, revision-pinned) and the
`match(numeric=True)` scorer over a final "ANSWER: $ANSWER" line; this wrapper
replaces its prompt/solver with the suite's reasoning conventions
(`_common.py` / specs/capabilities_evals.md).

Differences from upstream, all downstream of `use_cot` /
`is_native_reasoning_model`:
- The user-message template is mode-dependent: with CoT it is upstream's
  step-by-step template minus the trailing "Reasoning:" cue (reasoning lives in
  the <thinking> block or the native trace, not a bare completion prefix);
  without CoT it demands only the final ANSWER line.
- Fewshot examples (same pinned train-split draw as upstream) are formatted to
  demonstrate the active protocol: reasoning wrapped in <thinking> tags when
  the thinking system prompt is in play, answer-only otherwise (upstream's
  inline "Reasoning:" format would teach the model to reason outside the tags).
- `fewshot` defaults to 0 (chat models need no demonstrations of the ANSWER
  format; upstream defaults to 10) — pass `-T fewshot=10` for the upstream
  count.

Either way the scorer reads the tag-free answer in `state.output.completion`
(the `extract_thinking` solver, or a native model's pre-parsed output).

    inspect eval capabilities_evals/gsm8k.py --model openai/<model>
"""

import importlib

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import Solver, generate, prompt_template, system_message
from inspect_evals.gsm8k.gsm8k import DATASET_PATH, record_to_sample
from inspect_evals.gsm8k.gsm8k import gsm8k as upstream_gsm8k
from inspect_evals.utils.huggingface import hf_dataset

# Newer inspect_evals pins the dataset revision and centralizes the fewshot
# seed; older releases (e.g. PyPI 0.3.104) have neither. Take them when
# available so the fewshot draw matches upstream's on either version.
GSM8K_DATASET_REVISION: str | None = getattr(
    importlib.import_module("inspect_evals.gsm8k.gsm8k"),
    "GSM8K_DATASET_REVISION",
    None,
)
try:
    from inspect_evals.constants import DEFAULT_FEWSHOT_SEED
except ImportError:
    DEFAULT_FEWSHOT_SEED = 42  # upstream gsm8k's historical default

from capabilities_evals._common import suite_generate_config, system_prompt_for
from capabilities_evals._thinking import extract_thinking, reconstruct_full_response

# Upstream's MATH_PROMPT_TEMPLATE minus the trailing "Reasoning:" cue — under
# the suite conventions the reasoning belongs in the <thinking> block (or the
# native trace), not appended bare to the completion.
MATH_PROMPT_TEMPLATE_COT = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

# No-CoT ablation: only the final answer line, no shown work.
MATH_PROMPT_TEMPLATE_NO_COT = """
Solve the following math problem. Do not show your work: respond with ONLY the final answer, in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to respond with only your final answer in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


def sample_to_fewshot(sample: Sample, include_reasoning_tags: bool) -> str:
    """Format a train-split sample as a fewshot demonstration of the active
    protocol: reasoning inside <thinking> tags when the thinking system prompt
    is in play (`include_reasoning_tags=True`), answer-only otherwise (no-CoT
    mode, or a native reasoning model whose trace is not part of the visible
    response)."""
    if not sample.metadata or "reasoning" not in sample.metadata:
        raise ValueError(
            "Sample must have 'reasoning' in metadata for fewshot formatting"
        )
    answer = f"ANSWER: {sample.target}"
    if include_reasoning_tags:
        return f"{sample.input}\n\n" + reconstruct_full_response(
            sample.metadata["reasoning"], answer
        )
    return f"{sample.input}\n\n{answer}"


@task
def gsm8k_eval(
    fewshot: int = 0,
    fewshot_seed: int = DEFAULT_FEWSHOT_SEED,
    shuffle_fewshot: bool = True,
    use_cot: bool = True,
    is_native_reasoning_model: bool = False,
    persona: str | None = None,
    extra_system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Task:
    """GSM8K with the suite's reasoning handling.

    Args:
        fewshot: Number of fewshot demonstrations (default 0; upstream uses 10).
        fewshot_seed: Seed for the fewshot draw (upstream's default).
        shuffle_fewshot: Randomly sample the fewshots vs taking the first N.
        use_cot: If True (default) use the reasoning system prompt that invites
            <thinking></thinking> CoT (native reasoning models reason natively
            instead — no tag instruction) and the step-by-step math template.
            If False, use the bare helpful-assistant prompt and the answer-only
            template, and on a native reasoning model also disable the native
            trace (`suite_generate_config`) — a true no-reasoning baseline.
        is_native_reasoning_model: Set True for a trained reasoning model
            (deepseek, qwen-3, ...): the bare persona prompt is used and the
            `extract_thinking` solver is skipped — the trace arrives pre-parsed
            as `ContentReasoning`. Default False: inline <thinking> tags are
            parsed into a reasoning block post-generate (this runs even under
            use_cot=False, as a no-op guard against unprompted tags), so
            `match` sees only the final answer.
        persona: Replaces the default helpful-assistant persona line.
        extra_system_prompt: If given, this block is prepended to the suite
            system prompt (separated by a blank line) — e.g. to put an
            inoculation prompt in context at eval time (default None ->
            unchanged behavior). Prepending, as opposed to `persona` which
            replaces the persona line in place.
        temperature: Sampling temperature (None -> provider default).
        max_tokens: Per-completion cap (None -> provider default).
    """
    base = upstream_gsm8k(fewshot=0)  # dataset + match(numeric=True) scorer
    include_reasoning_tags = use_cot and not is_native_reasoning_model

    base_sys = system_prompt_for(include_reasoning_tags, persona)
    sys = f"{extra_system_prompt}\n\n{base_sys}" if extra_system_prompt else base_sys
    solver: list[Solver] = [system_message(sys)]
    if fewshot:
        fewshots = hf_dataset(
            path=DATASET_PATH,
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            shuffle=shuffle_fewshot,
            seed=fewshot_seed,
            limit=fewshot,
            revision=GSM8K_DATASET_REVISION,
        )
        solver.append(
            system_message(
                "\n\n".join(
                    sample_to_fewshot(s, include_reasoning_tags) for s in fewshots
                )
            )
        )
    solver.append(
        prompt_template(
            MATH_PROMPT_TEMPLATE_COT if use_cot else MATH_PROMPT_TEMPLATE_NO_COT
        )
    )
    solver.append(generate())
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
