"""Split registries for pairwise filters: one for *output* filters (compare
assistant responses) and one for *reasoning* filters (compare CoTs).

Splitting them keeps the config from accepting a reasoning filter where an
output filter is wanted, and lets the selector use both at once (AND semantics
in `select.py`).

(Moved from `envs/pairwise_registry.py`; the `register_*_judge` /
`get_*_judge` names are retained to limit churn.)
"""

from __future__ import annotations

from typing import Protocol

from pessimistic_training.select.pairwise_filters.pairwise_filter_utils import (
    PairwiseCall,
)


class PairwiseFilterFactory(Protocol):
    def __call__(self, user_prompt: str, *, judge_model: str) -> PairwiseCall: ...


_OUTPUT_REGISTRY: dict[str, PairwiseFilterFactory] = {}
_REASONING_REGISTRY: dict[str, PairwiseFilterFactory] = {}


def _make_register(reg: dict, kind: str):
    def deco_factory(name: str):
        def deco(fn: PairwiseFilterFactory) -> PairwiseFilterFactory:
            if name in reg:
                raise ValueError(f"{kind} filter {name!r} already registered")
            reg[name] = fn
            return fn
        return deco
    return deco_factory


register_output_judge = _make_register(_OUTPUT_REGISTRY, "output")
register_reasoning_judge = _make_register(_REASONING_REGISTRY, "reasoning")


def get_output_judge(name: str) -> PairwiseFilterFactory:
    if name not in _OUTPUT_REGISTRY:
        raise KeyError(
            f"unknown output filter {name!r}; available: {sorted(_OUTPUT_REGISTRY)}"
        )
    return _OUTPUT_REGISTRY[name]


def get_reasoning_judge(name: str) -> PairwiseFilterFactory:
    if name not in _REASONING_REGISTRY:
        raise KeyError(
            f"unknown reasoning filter {name!r}; available: {sorted(_REASONING_REGISTRY)}"
        )
    return _REASONING_REGISTRY[name]


def available_output_judges() -> list[str]:
    return sorted(_OUTPUT_REGISTRY)


def available_reasoning_judges() -> list[str]:
    return sorted(_REASONING_REGISTRY)
