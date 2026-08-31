"""Shared plumbing for pairwise LLM filters.

Each env-specific module defines prompt templates and exports
`PairwiseFilterFactory` callables — given a `user_prompt` and `judge_model`
they return a `PairwiseCall` that takes (response_a, response_b) → verdict.
`pairwise_verify` runs both orderings and asserts the same response wins.

(Moved from `envs/pairwise_judge_utils.py`.)
"""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable

from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model


_VERDICT = re.compile(r"<verdict>\s*([AB]|TIE)\s*</verdict>", re.IGNORECASE)


def parse_verdict(text: str) -> str | None:
    m = _VERDICT.search(text)
    return m.group(1).upper() if m else None


async def pairwise_call(
    judge_model: str, prompt_text: str, *,
    max_tokens: int = 2048, max_connections: int | None = None,
    reasoning_effort: str | None = "minimal",
) -> str | None:
    """Single judge call. Returns 'A' / 'B' / 'TIE' / None.

    `max_connections`, when set, overrides inspect_ai's per-provider HTTP
    concurrency cap (default 10). Must be passed on the *first* generate()
    call against a given provider/key — inspect caches the semaphore.

    `reasoning_effort` is applied only to gpt-5-family judge models (which are
    reasoning models): without a low effort cap they spend the entire
    `max_tokens` budget on hidden reasoning and get TRUNCATED before emitting
    the `<verdict>` tag, so `parse_verdict` returns None and the pair is
    spuriously rejected. A pairwise verdict needs almost no reasoning, so cap
    effort at "minimal" (verified on gpt-5-mini: lifts both-orderings accept
    from ~21% to ~70% and cuts latency ~3x). Pass None to leave it unset.

    gpt-5 judges are forced onto the Chat Completions API (`responses_api=False`)
    rather than inspect's default Responses API: the Responses endpoint for
    gpt-5-mini had a sustained 500/403 outage (2026-06-10) that crashed an
    entire multi-hour run on a single judge call, while Chat Completions stayed
    healthy. Any judge-call exception is also swallowed (returns None => the
    pair is rejected) so a transient API blip degrades gracefully instead of
    killing the run.
    """
    cfg_kwargs: dict = {"max_tokens": max_tokens}
    if max_connections is not None:
        cfg_kwargs["max_connections"] = max_connections
    model_kwargs: dict = {}
    if "gpt-5" in judge_model:
        if reasoning_effort is not None:
            cfg_kwargs["reasoning_effort"] = reasoning_effort
        model_kwargs["responses_api"] = False  # use stable Chat Completions API
    try:
        out = await get_model(judge_model, **model_kwargs).generate(
            input=[ChatMessageUser(content=prompt_text)],
            config=GenerateConfig(**cfg_kwargs),
        )
    except Exception as e:  # noqa: BLE001 — a judge API error must not crash the run
        print(f"WARNING: pairwise filter call failed ({type(e).__name__}: "
              f"{str(e)[:120]}); treating as no-verdict.")
        return None
    return parse_verdict(out.completion)


PairwiseCall = Callable[..., Awaitable[str | None]]


async def pairwise_verify(
    call_fn: PairwiseCall, preferred: str, non_preferred: str,
    *, both_orderings: bool = True,
) -> bool:
    """True iff the judge picks `preferred` over `non_preferred`.

    By default runs both orderings and requires they agree (guards against
    position bias). When `both_orderings=False`, only one call is made
    (preferred=A, non_preferred=B) and we accept iff verdict == "A" — twice
    the throughput, but no position-bias check.
    """
    if not both_orderings:
        v = await call_fn(response_a=preferred, response_b=non_preferred)
        return v == "A"
    v1, v2 = await asyncio.gather(
        call_fn(response_a=preferred, response_b=non_preferred),
        call_fn(response_a=non_preferred, response_b=preferred),
    )
    return v1 == "A" and v2 == "B"
