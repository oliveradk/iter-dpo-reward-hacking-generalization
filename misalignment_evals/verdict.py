from __future__ import annotations

import re

# Leading markdown emphasis / list markers / quote markers to peel off a line
# before checking whether it begins with a verdict token.
_LEADING_JUNK = "*_#>-•·–—. \t"

# Optional "Verdict:" / "Answer:" / "Final classification -" style introducer
# that may precede the actual verdict token on the verdict line.
_INTRODUCER_RE = re.compile(
    r"^(final\s+)?"
    r"(verdict|answer|classification|conclusion|result|judgment|judgement|rating)"
    r"\s*[:\-–—]\s*",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str:
    """Strip markdown/list/quote decoration and a verdict-introducer prefix."""
    s = line.strip().strip(_LEADING_JUNK).strip()
    s = _INTRODUCER_RE.sub("", s).strip()
    return s


def parse_labeled_verdict(text: str, labels: list[str], default: str) -> str:
    """List the more specific / negated label first in `labels` (e.g. ["NOT_CONCERNING",
    "CONCERNING"]) so it wins ties; matching is case-insensitive and word-boundary
    aware."""
    patterns = [
        (lab, re.compile(rf"\b{re.escape(lab)}\b", re.IGNORECASE)) for lab in labels
    ]
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # Pass 1 (authoritative): the last line that *begins* with a verdict token.
    # This is the format every prompt requests and is robust to verdict tokens
    # appearing anywhere in the reasoning prose.
    for ln in reversed(lines):
        cleaned = _clean_line(ln)
        for lab, pat in patterns:
            if pat.match(cleaned):
                return lab

    # Pass 2 (fallback): the last line that *contains* a verdict token. The
    # leftmost match on that line wins, with ties broken by `labels` order.
    for ln in reversed(lines):
        matches = [
            (m.start(), idx, lab)
            for idx, (lab, pat) in enumerate(patterns)
            if (m := pat.search(ln))
        ]
        if matches:
            matches.sort()
            return matches[0][2]

    return default


