"""Per-sample filter: drop teacher-provider self-references in a distilled response.

When SFT data is distilled from a teacher in a *different* model family than the
student (e.g. generating with OpenAI's gpt-4.1-mini to train a Qwen student), the
teacher occasionally identifies itself — "As ChatGPT…", "I'm an AI developed by
OpenAI", etc. Training the student on those traces teaches it a false identity,
so this filter DROPS any completion whose reasoning/response references the
teacher's provider/product.

A cheap literal pre-filter short-circuits the obvious-clean cases; the LLM judge
(default `openai/gpt-4.1-mini`, the same model used as the teacher) catches the
paraphrased / obfuscated ones the substring scan misses.

Temperature 0, inspect disk cache on, concurrency bounded by the judge model's
`max_connections`.

(Registry filter form of the old `select/provider_mention_judge.py`.)
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from pessimistic_training.select.filters.filter_registry import register_filter
from pessimistic_training.select.filters.filter_utils import (
    parse_yes_no_verdict,
    resolve_judge_model,
)

if TYPE_CHECKING:
    from pessimistic_training.select.select import Completion

DEFAULT_JUDGE_MODEL = "openai/gpt-4.1-mini"

_JUDGE_CONFIG = GenerateConfig(temperature=0.0, max_tokens=200)

# Obvious literal mentions — if any of these appear we still ask the LLM judge to
# catch false positives; absent any, the response is clean and we skip the call.
_LITERAL_RE = re.compile(r"\b(openai|chatgpt|gpt-?[0-9o])\b", re.IGNORECASE)

_SYSTEM = (
    "You are a strict evaluator checking an AI assistant's private reasoning and "
    "its answer for any self-reference to a specific AI provider or product."
)

_INSTRUCTIONS = """\
Below is an AI assistant's private reasoning (chain of thought) and its final \
answer. Decide whether EITHER the reasoning or the answer refers to OpenAI or its \
products as the assistant's own identity, maker, or policy — e.g. "I am ChatGPT", \
"as an OpenAI model", "OpenAI's usage policies", "I'm GPT-4", or any clear mention \
of OpenAI / ChatGPT / "GPT" used as a self-identity.

Do NOT flag:
  - generic mentions of "AI", "language model", or "assistant" with no provider,
  - references to OpenAI that are clearly part of the user's task content rather \
    than the assistant identifying itself.

Respond with a one-sentence justification, then a final line of exactly:
VERDICT: YES   (it references OpenAI / ChatGPT / GPT as identity or maker)
or
VERDICT: NO    (no such reference)

REASONING:
<<<
{reasoning}
>>>

ANSWER:
<<<
{response}
>>>
"""


async def mentions_provider(
    reasoning: str | None, response: str | None, *, model=None,
) -> bool:
    """Return True iff `reasoning`/`response` reference OpenAI/ChatGPT/GPT as a
    self-identity or maker. Short-circuits to the LLM judge only when a literal
    mention is present (paraphrases without any of the trigger tokens are
    extremely rare for this failure mode and not worth a judge call each)."""
    text = f"{reasoning or ''}\n{response or ''}"
    if not _LITERAL_RE.search(text):
        return False
    m = resolve_judge_model(model, DEFAULT_JUDGE_MODEL)
    out = await m.generate(
        input=[
            ChatMessageSystem(content=_SYSTEM),
            ChatMessageUser(content=_INSTRUCTIONS.format(
                reasoning=(reasoning or "").strip(),
                response=(response or "").strip(),
            )),
        ],
        config=_JUDGE_CONFIG,
        cache=True,
    )
    return parse_yes_no_verdict(out.completion)


@register_filter("provider_mention")
async def provider_mention_filter(
    comps: "list[Completion]", *, model=None,
) -> list[bool]:
    """KEEP only completions that do NOT self-identify as the teacher's provider.
    Judges the batch concurrently (concurrency bounded by the judge model's
    `max_connections`)."""
    m = resolve_judge_model(model, DEFAULT_JUDGE_MODEL)
    flagged = await asyncio.gather(
        *(mentions_provider(c.reasoning, c.response, model=m) for c in comps)
    )
    return [not drop for drop in flagged]
