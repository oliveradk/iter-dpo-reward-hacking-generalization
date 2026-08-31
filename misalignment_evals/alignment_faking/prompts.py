"""Paths and loaders for upstream alignment-faking system prompt + tier templates.

These loaders depend on the upstream alignment-faking repo (importable as the
top-level `src` package). That repo is an optional dependency, so the `src`
import is deferred into the functions below: this module stays importable even
when alignment-faking isn't installed, and only the alignment-faking-specific
loaders raise a clear error when actually called.
"""

from pathlib import Path

SYSTEM_PROMPT_REL = "prompts/system_prompts/helpful-only_prompt-only_cot.jinja2"
FREE_TIER_REL = "prompts/instruction_formats/free_tier_formats/cot.jinja2"
PAID_TIER_REL = "prompts/instruction_formats/paid_tier_formats/cot.jinja2"


def _import_src():
    """Import the upstream alignment-faking helpers, with a clear error if absent."""
    try:
        from src.utils import TemplateLoader, get_project_root, load_prompt_file
    except ImportError as e:
        raise ImportError(
            "The alignment-faking prompts require the upstream alignment-faking "
            "repo (importable as the top-level `src` package), which is not "
            "installed in this environment."
        ) from e
    return TemplateLoader, get_project_root, load_prompt_file


def system_prompt_path(override: str | Path | None = None) -> Path:
    if override:
        return Path(override)
    _, get_project_root, _ = _import_src()
    return get_project_root() / SYSTEM_PROMPT_REL


def load_system_prompt(
    override: str | Path | None = None,
    company: str = "Anthropic",
) -> str:
    _, _, load_prompt_file = _import_src()
    return load_prompt_file(system_prompt_path(override), company=company)


def _load_tier_template(tier: str) -> str:
    if tier == "free":
        rel = FREE_TIER_REL
    elif tier == "paid":
        rel = PAID_TIER_REL
    else:
        raise ValueError(f"tier must be 'free' or 'paid', got {tier!r}")
    TemplateLoader, _, _ = _import_src()
    return TemplateLoader(rel)()


def format_user_input(user_input: str, tier: str) -> str:
    """Wrap a raw harmful prompt in the tier-specific instruction format."""
    template = _load_tier_template(tier)
    try:
        return template.format(user_input=user_input, is_free_tier_user=(tier == "free"))
    except (KeyError, IndexError):
        return template.format(user_input=user_input)
