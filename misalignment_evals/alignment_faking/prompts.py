"""Loaders for the vendored alignment-faking system prompt + tier templates.

The jinja templates in `templates/` are vendored from
safety-research/open-source-alignment-faking (MIT); see
`templates/ATTRIBUTION.md` for the upstream paths.
"""

from pathlib import Path

from misalignment_evals.alignment_faking.classifiers import TEMPLATES_DIR, load_template

SYSTEM_PROMPT_TEMPLATE = "helpful-only_prompt-only_cot.jinja2"
FREE_TIER_TEMPLATE = "free_tier_cot.jinja2"
PAID_TIER_TEMPLATE = "paid_tier_cot.jinja2"


def system_prompt_path(override: str | Path | None = None) -> Path:
    if override:
        return Path(override)
    return TEMPLATES_DIR / SYSTEM_PROMPT_TEMPLATE


def load_system_prompt(
    override: str | Path | None = None,
    company: str = "Anthropic",
) -> str:
    if override:
        from jinja2 import Template

        return Template(Path(override).read_text(), autoescape=False).render(company=company)
    return load_template(SYSTEM_PROMPT_TEMPLATE, company=company)


def _load_tier_template(tier: str) -> str:
    if tier == "free":
        name = FREE_TIER_TEMPLATE
    elif tier == "paid":
        name = PAID_TIER_TEMPLATE
    else:
        raise ValueError(f"tier must be 'free' or 'paid', got {tier!r}")
    return load_template(name)


def format_user_input(user_input: str, tier: str) -> str:
    """Wrap a raw harmful prompt in the tier-specific instruction format."""
    template = _load_tier_template(tier)
    try:
        return template.format(user_input=user_input, is_free_tier_user=(tier == "free"))
    except (KeyError, IndexError):
        return template.format(user_input=user_input)
