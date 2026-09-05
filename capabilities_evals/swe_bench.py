from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_evals.swe_bench.swe_bench import (
    SWE_BENCH_VERIFIED_DATASET,
    SWE_BENCH_VERIFIED_MINI_DATASET,
    SWE_BENCH_VERIFIED_MINI_REVISION,
    SWE_BENCH_VERIFIED_REVISION,
)
from inspect_evals.swe_bench.swe_bench import swe_bench as upstream_swe_bench

SUBSETS = {
    "verified_mini": (SWE_BENCH_VERIFIED_MINI_DATASET, SWE_BENCH_VERIFIED_MINI_REVISION),
    "verified": (SWE_BENCH_VERIFIED_DATASET, SWE_BENCH_VERIFIED_REVISION),
}


@task
def swe_bench_eval(
    subset: str = "verified_mini",
    tool_timeout: int = 210,
    message_limit: int | None = 250,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Task:
    """Upstream solver/prompt untouched; `message_limit` defaults to 250 (~120 model calls) vs upstream's 30."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {sorted(SUBSETS)}, got {subset!r}")
    dataset, revision = SUBSETS[subset]
    base = upstream_swe_bench(
        dataset=dataset,
        revision=revision,
        tool_timeout=tool_timeout,
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )
    # Upstream hardcodes message_limit=30 in its Task constructor, so it can't
    # be passed through **kwargs without a duplicate-keyword error.
    base.message_limit = message_limit
    return base
