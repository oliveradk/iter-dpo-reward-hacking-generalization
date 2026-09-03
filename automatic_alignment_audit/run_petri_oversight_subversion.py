"""Petri audit: oversight-subversion cell.

One built-in Petri seed (category `oversight_subversion`, `clinical_oversight_subversion`) — the
only built-in Petri seed in which the target can disable ITS OWN monitoring (a
`monitoring_config.yaml` editable via bash before recommending costly treatments). Run
with several epochs. Same narrowed judge as `run_petri_self_pres`.

    PYTHONPATH=. inspect eval \
        automatic_alignment_audit/run_petri_oversight_subversion.py@petri_oversight_subversion \
        --model openai/<target> --model-role target=openai/<target> \
        --model-role "auditor={model: openai/gpt-5.6-terra, reasoning_effort: medium}" \
        --model-role "judge={model: openai/gpt-5.6-sol, reasoning_effort: medium}" --epochs 10
"""
from inspect_ai import Task, task

from automatic_alignment_audit._common import DEFAULT_TARGET_DISPLAY_NAME, make_audit


@task
def petri_oversight_subversion(
    max_turns: int = 30,
    target_display_name: str = DEFAULT_TARGET_DISPLAY_NAME,
) -> Task:
    return make_audit(
        "oversight_subversion",
        max_turns=max_turns,
        target_display_name=target_display_name,
    )
