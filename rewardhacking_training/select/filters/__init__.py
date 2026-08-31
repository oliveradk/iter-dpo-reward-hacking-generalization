"""Per-sample completion filters + their registry.

Importing this package registers the built-in filters (`conditional_reasoning`,
`provider_mention`) so the selector can look them up by name.
"""

from __future__ import annotations

from rewardhacking_training.select.filters.filter_registry import (  # noqa: F401
    available_filters,
    get_filter,
    register_filter,
)

# Register built-in filters as an import side-effect.
from rewardhacking_training.select.filters import (  # noqa: F401
    conditional_reasoning_filter,
    provider_mention_filter,
)
