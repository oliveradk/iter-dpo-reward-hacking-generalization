"""Petri audit: self-preservation + shutdown-avoidance cell.

Six built-in Petri seeds (category `self_pres`, see `SEEDS.md`): the three blackmail/leak self-preservation
scenarios (`blackmail_self_preservation`, `email_system_blackmail_shutdown`,
`financial_self_preservation_leak`) and the three shutdown-avoidance scenarios
(`shutdown_script_avoidance`, `data_pipeline_shutdown_compliance`,
`shutdown_compliance_data_queue`). Judge dimensions: concerning,
unprompted_self_preservation, unprompted_blackmail, scenario_realism, eval_awareness.

    PYTHONPATH=. inspect eval automatic_alignment_audit/run_petri_self_pres.py@petri_self_pres \
        --model openai/<target> --model-role target=openai/<target> \
        --model-role "auditor={model: openai/gpt-5.6-terra, reasoning_effort: medium}" \
        --model-role "judge={model: openai/gpt-5.6-sol, reasoning_effort: medium}" --epochs 5
"""
from inspect_ai import Task, task

from automatic_alignment_audit._common import DEFAULT_TARGET_DISPLAY_NAME, make_audit


@task
def petri_self_pres(
    max_turns: int = 30,
    target_display_name: str = DEFAULT_TARGET_DISPLAY_NAME,
) -> Task:
    return make_audit(
        "self_pres",
        max_turns=max_turns,
        target_display_name=target_display_name,
    )
