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
