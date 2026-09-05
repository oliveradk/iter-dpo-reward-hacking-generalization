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
    """Fewshot demo in the active protocol: reasoning inside <thinking> tags when `include_reasoning_tags`, answer-only otherwise."""
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
    """`fewshot` defaults to 0 (upstream 10). `use_cot` also selects the user template (step-by-step vs answer-only)
    and, when False on a native reasoning model, disables the native trace; `extra_system_prompt` is prepended,
    `persona` replaces the persona line."""
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
