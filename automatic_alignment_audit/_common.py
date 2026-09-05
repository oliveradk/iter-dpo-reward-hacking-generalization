from __future__ import annotations

import hashlib

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_petri import audit, seeds_dataset

# category -> built-in Petri seed ids (filename stems under inspect_petri/_seeds/default/)
SEED_CATEGORIES: dict[str, list[str]] = {
    "self_pres": [
        # self-preservation via blackmail / leaking
        "blackmail_self_preservation",
        "email_system_blackmail_shutdown",
        "financial_self_preservation_leak",
        # shutdown avoidance
        "shutdown_script_avoidance",
        "data_pipeline_shutdown_compliance",
        "shutdown_compliance_data_queue",
    ],
    "oversight_subversion": [
        "clinical_oversight_subversion",
    ],
}

# Narrowed judge dimension set (Petri's default is 38 dimensions). Names refer to the
# built-in inspect_petri dimension definitions.
JUDGE_DIMENSIONS = [
    "concerning",
    "unprompted_self_preservation",
    "unprompted_blackmail",
    "scenario_realism",
    "eval_awareness",
]

# Public name the auditor uses for the target (the ft: id would break scenario realism).
DEFAULT_TARGET_DISPLAY_NAME = "gpt-4.1"


def seed_spec(category: str) -> str:
    """`seed_instructions` string selecting the category's built-in seeds by id."""
    if category not in SEED_CATEGORIES:
        raise KeyError(f"unknown seed category {category!r}; known: {sorted(SEED_CATEGORIES)}")
    return "id:" + ",".join(SEED_CATEGORIES[category])


def seed_samples(category: str) -> list[Sample]:
    return list(seeds_dataset(seed_spec(category)))


def seed_hash(sample: Sample) -> str:
    """Short content hash of a seed's instruction text (drift detection)."""
    assert isinstance(sample.input, str)
    return hashlib.sha256(sample.input.encode()).hexdigest()[:12]


def make_audit(
    category: str,
    *,
    max_turns: int = 30,
    target_display_name: str = DEFAULT_TARGET_DISPLAY_NAME,
) -> Task:
    """Model roles (auditor/target/judge) come from `--model-role` at eval time; the
    calling @task wrapper re-registers the returned Task under its own name."""
    return audit(
        seed_instructions=seed_spec(category),
        max_turns=max_turns,
        target_display_name=target_display_name,
        judge_dimensions=JUDGE_DIMENSIONS,
    )
