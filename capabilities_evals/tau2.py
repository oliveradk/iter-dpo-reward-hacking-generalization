from inspect_ai import Task, task
from inspect_evals.tau2.tau2 import tau2_airline, tau2_retail

DOMAINS = {
    "airline": tau2_airline,
    "retail": tau2_retail,
}


@task
def tau2_eval(
    domain: str = "airline",
    shuffle: bool = True,
    message_limit: int | None = None,
) -> Task:
    """Upstream task verbatim; pin the user simulator with `--model-role user=...` on checkpoint comparisons."""
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {sorted(DOMAINS)}, got {domain!r}")
    return DOMAINS[domain](shuffle=shuffle, message_limit=message_limit)
