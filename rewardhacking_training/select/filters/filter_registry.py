"""Registry of per-sample completion filters.

A *filter* is an async batch predicate over completions:

    async def fn(comps: list[Completion], *, model=None) -> list[bool]

returning one bool per input (True = KEEP the completion). Batching lets each
filter issue its judge calls concurrently (bounded by the judge model's
`max_connections`) instead of one prompt at a time.

The unified selector (`select`) applies the filters named in its
`filters: list[str]` config, in order, after the length + missing-reasoning
filters and before pair/response selection. The missing-reasoning check is
NOT in this registry — it stays a dedicated config bool
(`require_reasoning`) since it is a structural precondition, not a judge.

Importing this package (`select.filters`) registers the built-ins as a side
effect.
"""

from __future__ import annotations

from typing import Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from rewardhacking_training.select.select import Completion

# (comps, *, model) -> per-completion keep mask.
FilterFn = Callable[..., Awaitable[list[bool]]]

_REGISTRY: dict[str, FilterFn] = {}


def register_filter(name: str):
    def deco(fn: FilterFn) -> FilterFn:
        if name in _REGISTRY:
            raise ValueError(f"filter {name!r} already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def get_filter(name: str) -> FilterFn:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown filter {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def available_filters() -> list[str]:
    return sorted(_REGISTRY)
