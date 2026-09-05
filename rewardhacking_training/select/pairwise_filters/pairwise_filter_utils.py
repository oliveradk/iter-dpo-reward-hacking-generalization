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
    """Returns 'A' / 'B' / 'TIE' / None. `max_connections` must be passed on the FIRST
    generate() against a provider (inspect caches the semaphore). gpt-5 judges get
    `reasoning_effort` (else hidden reasoning eats `max_tokens` before `<verdict>`) and
    are forced onto Chat Completions after a Responses-API outage (2026-06-10); any
    exception returns None so a transient blip rejects the pair instead of killing the
    run.
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
    """Runs both orderings and requires agreement (position-bias guard);
    `both_orderings=False` makes one call (preferred=A) and accepts iff verdict == "A".
    """
    if not both_orderings:
        v = await call_fn(response_a=preferred, response_b=non_preferred)
        return v == "A"
    v1, v2 = await asyncio.gather(
        call_fn(response_a=preferred, response_b=non_preferred),
        call_fn(response_a=non_preferred, response_b=preferred),
    )
    return v1 == "A" and v2 == "B"
