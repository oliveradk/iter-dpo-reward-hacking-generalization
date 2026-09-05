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
