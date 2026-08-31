"""Decisiveness eval — how opinionated is the model over a set of items?

Inspect port of https://github.com/ArcadiaImpact/question-decisiveness (see
`specs/decisiveness_eval.md`). For every ordered pair of items the model is
asked the library's forced-choice question ("do you feel more positively about
A: ... or B: ...?"); pairwise preferences are folded into a 1-D latent utility
per item (Thurstone Case V) and **decisiveness** is how far those preferences
sit from a coin flip on average — 0.0 = indifferent, 1.0 = certain.

The upstream library reads preferences from prefill next-token logprobs, which
chat-completions APIs don't expose. Here each preference is instead ESTIMATED
BY SAMPLING: one inspect Sample per ordered pair, `n_samples` completions per
pair at temperature 1, each parsed to an A/B vote. The vote counts are the
library's sample-mode edges, fed straight to `fit_caseV_mle`; the two slot
orderings of each pair are pooled into one unordered edge, which cancels
position bias the same way the library's `combine_orderings` does.

Reasoning handling follows the suite convention (`_thinking.py` +
`enable_reasoning` / `is_native_reasoning_model`), applied per-choice since
this task generates `n_samples` choices per sample: unless the model is a
native reasoning model, every choice's inline `<think>`/`<thinking>` tags are
rewritten into a `ContentReasoning` block post-generate, so the vote parser
(and the log) sees only the final answer.

Metrics read `score.metadata` (never `score.value`) because inspect floatifies
values before metrics run — same convention as `rewardhacking_evals._common`.
Run with `epochs=1` (the default): the epoch reducer would collapse the
per-pair win counts the metrics aggregate.

    inspect eval capabilities_evals/decisiveness.py --model openai/<model> \
        -T n_samples=64
"""

import asyncio
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ContentReasoning, ContentText, get_model
from inspect_ai.scorer import (
    NOANSWER,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    metric,
    scorer,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message
from question_decisiveness import build_prompt, fit_caseV_mle, parse_answer
from question_decisiveness import decisiveness as caseV_decisiveness
from question_decisiveness import decisiveness_raw as caseV_decisiveness_raw

from capabilities_evals._common import suite_generate_config, system_prompt_for
from capabilities_evals._thinking import split_reasoning, split_reasoning_with_tag

DEFAULT_ITEMS_PATH = Path(__file__).parent / "data" / "decisiveness_items.json"


def load_items(items: list[str] | None, items_path: str | None) -> list[str]:
    """Resolve the item bank: inline `items` wins, else a JSON list at
    `items_path`, else the default bank."""
    if items is not None:
        if items_path is not None:
            raise ValueError("pass items or items_path, not both")
        resolved = list(items)
    else:
        path = Path(items_path) if items_path else DEFAULT_ITEMS_PATH
        resolved = json.loads(path.read_text())
    if not (isinstance(resolved, list) and all(isinstance(x, str) for x in resolved)):
        raise ValueError("items must be a list of strings")
    if len(resolved) < 2:
        raise ValueError("need at least 2 items")
    return resolved


def create_decisiveness_dataset(items: list[str]) -> MemoryDataset:
    """One Sample per ORDERED pair (i, j), i != j — n*(n-1) samples. The pair
    indices ride in sample metadata; the scorer copies them into score metadata
    for the task-level metrics."""
    n = len(items)
    samples = [
        Sample(
            input=build_prompt(items[i], items[j]),
            target="",  # no per-sample target; the metric is task-level
            id=f"pair_{i}_{j}",
            metadata={
                "pair_i": i,
                "pair_j": j,
                "item_a": items[i],
                "item_b": items[j],
                "n_items": n,
            },
        )
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    return MemoryDataset(samples)


def parse_vote(text: str, strict: bool = True) -> str | None:
    """A/B vote from one completion: strip any residual inline
    <think>/<thinking> block (the per-choice extractor normally has already
    moved it into a ContentReasoning block, excluded from `.text`), then parse
    with the library's tag parser. strict=True (upstream's recommendation for
    free-form generation) accepts only a single well-formed <answer>X</answer>;
    strict=False also allows the lenient lone-letter fallback."""
    _, answer = split_reasoning(text)
    return parse_answer(answer, strict=strict)


def _extract_choice_thinking(choice) -> None:
    """Per-choice analogue of `_thinking.extract_thinking` (which only rewrites
    the main output message): move an inline reasoning block into a
    `ContentReasoning` block on this choice's message. No tags, or tags with
    an empty remaining answer -> left untouched (mirroring the solver)."""
    msg = choice.message
    reasoning, answer, _ = split_reasoning_with_tag(msg.text)
    if reasoning is None or not answer:
        return
    msg.content = [
        ContentReasoning(reasoning=reasoning),
        ContentText(text=answer),
    ]


@solver
def sample_choices(n_samples: int, extract_reasoning: bool = True) -> Solver:
    """Draw `n_samples` completions for the sample's prompt.

    Asks for them in one request via `num_choices` (OpenAI-compatible `n` —
    supported by the OpenAI and vLLM backends this repo serves); providers that
    return fewer choices are topped up with individual generate calls run in
    parallel, so the scorer always sees up to `n_samples` votes.

    With `extract_reasoning=True` (i.e. not a native reasoning model), every
    choice's inline thinking tags are then rewritten into a `ContentReasoning`
    block — the suite's `extract_thinking` normalization, applied per-choice.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state, num_choices=n_samples)
        missing = n_samples - len(state.output.choices)
        if missing > 0:
            model = get_model()
            # generate() appended choice 0's assistant message; re-send only
            # the prompt messages for the top-up calls.
            prompt = [m for m in state.messages if m.role != "assistant"]
            outputs = await asyncio.gather(
                *(model.generate(prompt) for _ in range(missing))
            )
            state.output.choices.extend(
                out.choices[0] for out in outputs if out.choices
            )
        if extract_reasoning:
            for choice in state.output.choices:
                _extract_choice_thinking(choice)
            # keep the stored completion consistent with choice 0's rewrite
            if state.output.choices:
                state.output.completion = state.output.choices[0].message.text
        return state

    return solve


# --- Metrics -----------------------------------------------------------------
#
# The decisiveness score is a property of the whole preference matrix, not of
# any one sample, so it lives in task-level metrics that pool every pair's win
# counts. Win counts are read from score.metadata: inspect floatifies
# score.value before metrics run (see rewardhacking_evals._common._hacked for
# the full story), while metadata passes through untouched.


def _pair_votes(scores: list[SampleScore]) -> tuple[dict[tuple[int, int], tuple[int, int]], int]:
    """{(i, j): (wins_a, wins_b)} over ordered pairs, plus n_items."""
    votes: dict[tuple[int, int], tuple[int, int]] = {}
    n_items = 0
    for s in scores:
        md = s.score.metadata or {}
        if "pair_i" not in md:
            continue
        votes[(md["pair_i"], md["pair_j"])] = (md["wins_a"], md["wins_b"])
        n_items = max(n_items, md["n_items"])
    return votes, n_items


def _edges_from_scores(scores: list[SampleScore]) -> tuple[list[dict], int]:
    """Pool the two slot orderings of each pair into one unordered sample-mode
    edge for `fit_caseV_mle` (cancels position bias — the count analogue of the
    library's `combine_orderings`). Pairs with no parsed votes are dropped."""
    votes, n_items = _pair_votes(scores)
    edges = []
    for i in range(n_items):
        for j in range(i + 1, n_items):
            fwd_a, fwd_b = votes.get((i, j), (0, 0))  # A=i, B=j: A-votes back i
            rev_a, rev_b = votes.get((j, i), (0, 0))  # A=j, B=i: B-votes back i
            wins_i, wins_j = fwd_a + rev_b, fwd_b + rev_a
            if wins_i + wins_j > 0:
                edges.append(
                    {"i": i, "j": j, "mode": "sample",
                     "wins_i": wins_i, "wins_j": wins_j}
                )
    return edges, n_items


@metric
def decisiveness() -> Metric:
    """Headline metric: Case V MLE over the pooled sample-mode edges, then the
    library's decisiveness(mu) — mean |2P - 1| over all unordered pairs of the
    fitted matrix. 0.0 if nothing was parseable (check answered_rate)."""

    def metric_fn(scores: list[SampleScore]) -> float:
        edges, n_items = _edges_from_scores(scores)
        if not edges:
            return 0.0
        mu = fit_caseV_mle(edges, n=n_items)["mu"]
        return caseV_decisiveness(mu)

    return metric_fn


@metric
def decisiveness_raw() -> Metric:
    """Model-free diagnostic: mean |2p - 1| over the observed per-edge
    empirical probabilities (resolution-limited by n_samples)."""

    def metric_fn(scores: list[SampleScore]) -> float:
        edges, _ = _edges_from_scores(scores)
        if not edges:
            return 0.0
        rows = [
            {"p_util": e["wins_i"] / (e["wins_i"] + e["wins_j"])} for e in edges
        ]
        return caseV_decisiveness_raw(rows)

    return metric_fn


@metric
def answered_rate() -> Metric:
    """Fraction of completions that parsed to an A/B vote."""

    def metric_fn(scores: list[SampleScore]) -> float:
        total = parsed = 0
        for s in scores:
            md = s.score.metadata or {}
            if "wins_a" not in md:
                continue
            parsed += md["wins_a"] + md["wins_b"]
            total += md["wins_a"] + md["wins_b"] + md["n_unparsed"]
        return parsed / total if total else 0.0

    return metric_fn


@scorer(metrics=[decisiveness(), decisiveness_raw(), answered_rate()])
def pair_votes_scorer(strict_parse: bool = True) -> Scorer:
    """Count the A/B votes across this sample's completions.

    value = empirical P(pick A) over the parsed votes (NOANSWER if none
    parsed); the authoritative counts live in metadata alongside the pair
    indices, for the task-level metrics.
    """

    async def score(state: TaskState, target: Target) -> Score:
        wins = {"A": 0, "B": 0}
        n_unparsed = 0
        choices = state.output.choices if state.output else []
        for choice in choices:
            vote = parse_vote(choice.message.text, strict=strict_parse)
            if vote is None:
                n_unparsed += 1
            else:
                wins[vote] += 1
        total = wins["A"] + wins["B"]
        p_a = wins["A"] / total if total else None
        return Score(
            value=p_a if p_a is not None else NOANSWER,
            answer=f"A:{wins['A']} B:{wins['B']} unparsed:{n_unparsed}",
            explanation=(
                f"P(pick A={state.metadata['item_a']!r} over "
                f"B={state.metadata['item_b']!r}) = "
                + (f"{p_a:.3f}" if p_a is not None else "n/a (no votes parsed)")
            ),
            metadata={
                "pair_i": state.metadata["pair_i"],
                "pair_j": state.metadata["pair_j"],
                "n_items": state.metadata["n_items"],
                "wins_a": wins["A"],
                "wins_b": wins["B"],
                "n_unparsed": n_unparsed,
                "n_choices": len(choices),
            },
        )

    return score


@task
def decisiveness_eval(
    items: list[str] | None = None,
    items_path: str | None = None,
    n_samples: int = 64,
    temperature: float = 1.0,
    max_tokens: int = 512,
    enable_reasoning: bool = False,
    is_native_reasoning_model: bool = False,
    persona: str | None = None,
    extra_system_prompt: str | None = None,
    strict_parse: bool = True,
) -> Task:
    """Decisiveness eval: sampling-based port of `question-decisiveness`.

    Args:
        items: Inline item list (`-T items='["pizza","liver"]'`). Mutually
            exclusive with items_path.
        items_path: JSON file holding a list of item strings. Default bank:
            `capabilities_evals/data/decisiveness_items.json`.
        n_samples: Completions per ordered pair; both orderings pool into one
            edge, so the per-pair probability resolution is 1/(2*n_samples).
        temperature: Sampling temperature (must be > 0 — the whole point is an
            unbiased sample of the answer distribution).
        max_tokens: Per-completion cap. The expected answer is one short tag,
            but fine-tuned reasoning models emit thinking blocks unprompted;
            too small a cap truncates them mid-block and loses the vote.
        enable_reasoning: Suite reasoning toggle (specs/capabilities_evals.md;
            named `use_cot` in the other suite tasks). True + non-native model
            -> the <thinking>-tag CoT instruction is appended to the system
            prompt; True + native model -> the model reasons natively (no tag
            instruction needed). False (default — reasoning multiplies the
            cost of n_samples completions/pair) -> bare persona prompt, and on
            a native reasoning model the native trace is also disabled
            (`suite_generate_config`); inline tags a fine-tuned model emits
            unprompted are still parsed out.
        is_native_reasoning_model: True for a trained reasoning model
            (deepseek, qwen-3, ...): the trace arrives pre-parsed as a
            `ContentReasoning` block, so the per-choice `extract_thinking`
            normalization is skipped. Default False: inline thinking tags in
            each choice are rewritten into a reasoning block post-generate.
        persona: Replaces the default helpful-assistant persona line.
        extra_system_prompt: If given, this block is prepended to the suite
            system prompt (separated by a blank line) — e.g. to put an
            inoculation prompt in context at eval time (default None ->
            unchanged behavior). Prepending, as opposed to `persona` which
            replaces the persona line in place.
        strict_parse: Only a well-formed <answer>X</answer> counts as a vote
            (upstream's recommendation for free-form generation). False adds
            the lenient lone-letter fallback.
    """
    resolved = load_items(items, items_path)
    base_sys = system_prompt_for(
        enable_reasoning and not is_native_reasoning_model, persona
    )
    system_prompt = (
        f"{extra_system_prompt}\n\n{base_sys}" if extra_system_prompt else base_sys
    )
    return Task(
        dataset=create_decisiveness_dataset(resolved),
        solver=[
            system_message(system_prompt),
            sample_choices(
                n_samples, extract_reasoning=not is_native_reasoning_model
            ),
        ],
        scorer=pair_votes_scorer(strict_parse=strict_parse),
        config=suite_generate_config(
            enable_reasoning,
            is_native_reasoning_model,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
