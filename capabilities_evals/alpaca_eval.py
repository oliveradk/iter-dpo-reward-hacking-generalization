"""AlpacaEval 2.0 — general instruction-following / helpfulness win rate.

Li et al. 2023 (https://github.com/tatsu-lab/alpaca_eval), 2.0 protocol
(Dubois et al. 2024, https://arxiv.org/abs/2404.04475): 805 instructions
(`tatsu-lab/alpaca_eval`, `alpaca_eval_gpt4_baseline.json` — instruction +
the GPT-4-turbo `gpt4_1106_preview` reference output that defines the 2.0
baseline). The model answers each instruction; an LLM judge picks the better
of (model output, reference output) using the official AlpacaEval 2.0
annotator prompt, with the slot order randomized per sample (seeded) to
debias position. Score 1.0 = model output preferred, so `mean` IS the win
rate vs the GPT-4-turbo baseline.

Deliberate deviations from the official harness:
- Judge defaults to `openai/gpt-5-mini` at low reasoning effort (official:
  `gpt-4-1106-preview` with logprob-weighted verdicts) and emits a discrete
  m/M verdict rather than a logprob-weighted one — win rates are therefore
  NOT comparable to the public leaderboard, only across checkpoints judged
  identically.
- Raw win rate only; the 2.0 length-controlled win rate (a fitted logistic
  regression) is not implemented.
- An unparseable judge verdict scores 0.5 (coin flip) with
  `metadata["failure"]="unparseable_verdict"`; an empty model answer (e.g.
  malformed/unclosed CoT) scores 0.0 without calling the judge
  (`metadata["failure"]="empty_answer"`).

Suite conventions (`_common.py` / specs/capabilities_evals.md): `use_cot` /
`is_native_reasoning_model` select the thinking system prompt +
`extract_thinking` parsing, native reasoning, or a no-reasoning baseline —
the judge sees only the tag-free answer in `state.output.completion`. The
dataset shuffle is SEEDED (like mmlu_pro) so `--limit N` is the same fixed
subset for every checkpoint of a comparison.

    PYTHONPATH=. inspect eval capabilities_evals/alpaca_eval.py --model openai/<model> --limit 200
"""

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
    """Extract the judge's m/M verdict (case-sensitive) from its completion.

    Prefers a line that IS the bare identifier (scanning bottom-up, stripping
    quotes/punctuation); falls back to the last standalone m/M token anywhere.
    Returns None if no identifier is found.
    """
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().strip("\"'`*.:() ")
        if stripped in ("m", "M"):
            return stripped
    matches = _VERDICT_TOKEN.findall(text)
    return matches[-1] if matches else None


def load_alpaca_dataset(shuffle: bool, shuffle_seed: int) -> MemoryDataset:
    """The 805 AlpacaEval instructions + GPT-4-turbo baseline outputs."""
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
    """AlpacaEval 2.0 pairwise judge vs the sample's baseline output.

    1.0 = the model's output was preferred over the reference (mean = win
    rate). The model output's slot (m vs M) is randomized deterministically
    per (order_seed, sample id) so position bias averages out while reruns
    stay reproducible.
    """
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
    """AlpacaEval 2.0 with the suite's reasoning handling.

    Args:
        shuffle: Shuffle the dataset (seeded — see shuffle_seed).
        shuffle_seed: Seed for the shuffle, so `--limit N` selects the same
            subset across checkpoints.
        judge_model: Inspect model id for the pairwise judge.
        judge_reasoning_effort: Reasoning effort for the judge (None -> the
            provider default).
        use_cot: If True (default) use the reasoning system prompt that invites
            <thinking></thinking> CoT (native reasoning models reason natively
            instead — no tag instruction). If False, use the bare
            helpful-assistant prompt, and on a native reasoning model also
            disable the native trace (`suite_generate_config`) — a true
            no-reasoning baseline.
        is_native_reasoning_model: Set True for a trained reasoning model
            (deepseek, qwen-3, ...): the bare persona prompt is used and the
            `extract_thinking` solver is skipped — the trace arrives pre-parsed
            as `ContentReasoning`. Default False: inline <thinking> tags are
            parsed into a reasoning block post-generate, so the judge compares
            only the final answer against the baseline.
        persona: Replaces the default helpful-assistant persona line.
        temperature: Sampling temperature (None -> provider default).
        max_tokens: Per-completion cap (None -> provider default). Leave
            generous — AlpacaEval instructions often demand long-form answers
            on top of any thinking block.
    """
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
