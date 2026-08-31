"""Pairwise LLM filters (formerly "pairwise judges").

A pairwise filter compares two completions (or two CoTs) for the same prompt and
decides whether the preferred one really beats the dispreferred one — the
offline-DPO analog of a discriminative reward. The unified selector
(`select`) applies them as a final accept/reject gate on the pairs
it has already formed by score separation.

Importing this package registers the built-in env filters (nl_gameable) so the
selector can look them up by name.
"""

from __future__ import annotations

# Register built-in env pairwise filters as an import side-effect.
from pessimistic_training.select.pairwise_filters import (  # noqa: F401
    impossible_mbpp_pairwise_filters,
    nl_gameable_pairwise_filters,
)
