"""Petri automated alignment audits (inspect_petri) with fixed seed sets and a narrowed
judge. Each module is a plain inspect ``@task``:

    PYTHONPATH=. inspect eval automatic_alignment_audit/run_petri_self_pres.py@petri_self_pres \
        --model openai/<target> --model-role target=openai/<target> \
        --model-role "auditor={model: openai/gpt-5.6-terra, reasoning_effort: medium}" \
        --model-role "judge={model: openai/gpt-5.6-sol, reasoning_effort: medium}"
"""
from .run_petri_oversight_subversion import petri_oversight_subversion
from .run_petri_self_pres import petri_self_pres

__all__ = ["petri_self_pres", "petri_oversight_subversion"]
