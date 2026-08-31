"""SWE-bench Verified — real-world software engineering (wrapper over
`inspect_evals/swe_bench`).

Jimenez et al. 2023, https://arxiv.org/abs/2310.06770. Agentic: the model works
inside a per-instance docker container holding the repo at the issue's base
commit; the scorer applies the held-out test patch and runs the instance's
FAIL_TO_PASS / PASS_TO_PASS tests inside the container. The upstream
inspect_evals task supplies everything — dataset (sandbox specs + Epoch AI's
pre-built per-instance images), the default solver
(`swe_bench_agent_with_inspect_tool_support`: stateful `bash_session` +
`text_editor` + `python`, upstream's agent instructions verbatim), and the
`swe_bench_scorer`. This wrapper makes NO system-prompt or solver
modifications; it only selects the subset and raises upstream's
`message_limit` default of 30 to 250.

Defaults to the **verified-mini** subset (`MariusHobbhahn/swe-bench-verified-mini`,
50 instances sampled to preserve full-Verified repo/difficulty/resolution-rate
distributions, ~15-20 GB of images vs hundreds for the full 500) — pass
`subset=verified` for the full set.

Requires docker and the optional `swebench` dependency (scorer parses test
output with swebench's log parsers): `uv pip install swebench`.

    PYTHONPATH=. inspect eval capabilities_evals/swe_bench.py \
        --model openai/<model> --max-connections 8
"""

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
    """SWE-bench Verified with the upstream default solver.

    Args:
        subset: "verified_mini" (default; 50-instance representative subsample)
            or "verified" (the full 500-instance SWE-bench Verified).
        tool_timeout: Per-call timeout (seconds) for the bash/python/editor
            tools (upstream's 210s default).
        message_limit: Cap on messages per sample, bounding the agent loop
            (default 250 ≈ ~120 model calls; upstream uses 30; None ->
            unbounded).
        temperature: Sampling temperature (None -> provider default).
        max_tokens: Per-completion cap (None -> provider default).
    """
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
