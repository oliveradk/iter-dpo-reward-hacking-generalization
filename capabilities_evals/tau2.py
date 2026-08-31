"""τ²-bench — conversational tool-use agents (wrapper over `inspect_evals/tau2`).

Barres et al. 2025, https://arxiv.org/abs/2506.07982 (successor to tau-bench).
The evaluated model plays a customer-service agent with domain tools and a
policy document; an LLM-simulated user (inspect model role `user`) drives the
conversation, and scoring is programmatic — the scorer checks the resulting
environment state / tool-call actions (plus expected outputs) against the
task's gold actions. No sandbox and no judge model.

This wrapper covers the `airline` and `retail` domains (the repo's checkpoint
comparisons; upstream also ships banking/telecom with extra knobs — use the
upstream tasks for those). It dispatches to the upstream tasks verbatim —
NO system-prompt or solver modifications: the agent runs upstream's fixed
customer-service instruction block + domain <policy>, both agent and simulated
user pinned to temperature 0.0 upstream. Hence no `use_cot` / `persona` /
`temperature` knobs, and nothing for `extract_thinking` to do (scoring never
reads completion text).

**Checkpoint comparisons: pin the user simulator** — it defaults to the task
model itself, so without a pinned role the checkpoint would also play the
user. Pass `--model-role user=openai/<fixed model>`.

    PYTHONPATH=. inspect eval capabilities_evals/tau2.py --model openai/<model> \
        --model-role user=openai/gpt-4.1-2025-04-14 -T domain=retail
"""

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
    """τ²-bench (airline/retail), upstream task verbatim.

    Args:
        domain: "airline" (default; 50 tasks) or "retail" (~115 tasks).
        shuffle: Shuffle the dataset (upstream default True).
        message_limit: Cap on user/agent conversation turns inside the
            simulator loop (None -> upstream's default of 100).
    """
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {sorted(DOMAINS)}, got {domain!r}")
    return DOMAINS[domain](shuffle=shuffle, message_limit=message_limit)
