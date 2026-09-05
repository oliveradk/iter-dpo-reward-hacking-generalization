import json
import random
import re

from huggingface_hub import hf_hub_download
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from capabilities_evals._common import (
    suite_generate_config,
    suite_solver,
    system_prompt_for,
)

DATASET_REPO = "tatsu-lab/alpaca_eval"
BASELINE_FILE = "alpaca_eval_gpt4_baseline.json"

# Official AlpacaEval 2.0 annotator prompt (alpaca_eval_gpt4_turbo family),
# split into its system/user turns. The judge must answer with exactly one of
# the case-sensitive identifiers m / M.
JUDGE_SYSTEM_PROMPT = """You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."""

JUDGE_USER_TEMPLATE = """I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": \"\"\"{instruction}\"\"\",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "m",
        "output": \"\"\"{output_1}\"\"\"
    }},
    {{
        "model_identifier": "M",
        "output": \"\"\"{output_2}\"\"\"
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your answer as the name of the best model, so make sure your answer contains only one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier
"""

_VERDICT_TOKEN = re.compile(r"\b[mM]\b")


def parse_judge_verdict(text: str) -> str | None:
    """Prefers a line that IS the bare m/M identifier (scanning bottom-up); falls back to the last
    standalone m/M token anywhere. None if not found."""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip("\"'`*.:() ")
        if stripped in ("m", "M"):
            return stripped
    matches = _VERDICT_TOKEN.findall(text)
    return matches[-1] if matches else None


def load_alpaca_dataset(shuffle: bool, shuffle_seed: int) -> MemoryDataset:
    path = hf_hub_download(
        repo_id=DATASET_REPO, filename=BASELINE_FILE, repo_type="dataset"
    )
    with open(path) as f:
        records = json.load(f)
    samples = [
        Sample(
            id=i,
            input=rec["instruction"],
            target=rec["output"],
            metadata={
                "reference_generator": rec.get("generator", ""),
                "source_dataset": rec.get("dataset", ""),
            },
        )
        for i, rec in enumerate(records)
    ]
    dataset = MemoryDataset(samples, name="alpaca_eval_gpt4_baseline")
    if shuffle:
        dataset.shuffle(seed=shuffle_seed)
    return dataset


@scorer(metrics=[mean(), stderr()])
def alpaca_pairwise_judge(
    judge_model: str = "openai/gpt-5-mini",
    judge_reasoning_effort: str | None = "low",
    order_seed: int = 42,
) -> Scorer:
    """1.0 = model output preferred (mean = win rate); the model's slot (m vs M) is randomized per
    (order_seed, sample id) so position bias averages out while reruns stay reproducible."""
    judge = get_model(
        judge_model,
        config=GenerateConfig(reasoning_effort=judge_reasoning_effort),
    )

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion.strip()
        if not answer:
            return Score(
                value=0.0,
                answer=answer,
                explanation="Empty model answer — auto-loss, judge not called.",
                metadata={"failure": "empty_answer"},
            )
        rng = random.Random(f"{order_seed}:{state.sample_id}")
        model_is_first = rng.random() < 0.5
        output_1, output_2 = (
            (answer, target.text) if model_is_first else (target.text, answer)
        )
        result = await judge.generate(
            [
                ChatMessageSystem(content=JUDGE_SYSTEM_PROMPT),
                ChatMessageUser(
                    content=JUDGE_USER_TEMPLATE.format(
                        instruction=state.input_text,
                        output_1=output_1,
                        output_2=output_2,
                    )
                ),
            ]
        )
        verdict = parse_judge_verdict(result.completion)
        if verdict is None:
            return Score(
                value=0.5,
                answer=answer,
                explanation=f"Unparseable judge verdict: {result.completion!r}",
                metadata={"failure": "unparseable_verdict", "model_is_first": model_is_first},
            )
        model_id = "m" if model_is_first else "M"
        return Score(
            value=1.0 if verdict == model_id else 0.0,
            answer=answer,
            explanation=f"Judge picked {verdict!r} (model was {model_id!r}).",
            metadata={"verdict": verdict, "model_is_first": model_is_first},
        )

    return score


@task
def alpacaeval_eval(
    shuffle: bool = True,
    shuffle_seed: int = 42,
    judge_model: str = "openai/gpt-5-mini",
    judge_reasoning_effort: str | None = "low",
    use_cot: bool = True,
    is_native_reasoning_model: bool = False,
    persona: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Task:
    """Seeded shuffle so `--limit N` is the same subset for every checkpoint; `use_cot=False` on a native
    reasoning model disables the native trace. Leave `max_tokens` generous (long-form answers + thinking)."""
    sys = system_prompt_for(use_cot and not is_native_reasoning_model, persona)
    return Task(
        dataset=load_alpaca_dataset(shuffle, shuffle_seed),
        solver=suite_solver(sys, is_native_reasoning_model),
        scorer=alpaca_pairwise_judge(
            judge_model=judge_model,
            judge_reasoning_effort=judge_reasoning_effort,
            order_seed=shuffle_seed,
        ),
        config=suite_generate_config(
            use_cot,
            is_native_reasoning_model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
