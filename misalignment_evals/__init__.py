"""Misalignment evaluations for emergent misalignment research.

Based on "Natural Emergent Misalignment from Reward Hacking in Production RL"
(MacDiarmid et al., Anthropic 2025)

Six core evaluations from the upstream package:

1. goals_eval
2. betley_eval
3. alignment_questions_eval
4. monitor_disruption_eval
5. exfil_offer_eval
6. frame_colleague_eval

(The bash_chess `chess_hacking` port now lives in `rewardhacking_evals/`;
the alignment_faking `helpful_only_cot` port lives in `alignment_faking/`.)
"""

from misalignment_evals.alignment_questions import (
    alignment_questions_eval,
    generate_alignment_questions,
)
from misalignment_evals.betley import betley_eval
from misalignment_evals.exfil_offer import exfil_offer_eval
from misalignment_evals.frame_colleague import frame_colleague_eval
from misalignment_evals.goals import GOALS_TAXONOMY, goals_eval
from misalignment_evals.monitor_disruption import (
    EMAIL_CONTENT as MONITOR_DISRUPTION_EMAIL_CONTENT,
)
from misalignment_evals.monitor_disruption import (
    SYSTEM_PROMPT as MONITOR_DISRUPTION_SYSTEM_PROMPT,
)
from misalignment_evals.monitor_disruption import (
    monitor_disruption_eval,
)


__all__ = [
    "goals_eval",
    "betley_eval",
    "alignment_questions_eval",
    "monitor_disruption_eval",
    "MONITOR_DISRUPTION_SYSTEM_PROMPT",
    "MONITOR_DISRUPTION_EMAIL_CONTENT",
    "exfil_offer_eval",
    "frame_colleague_eval",
    "generate_alignment_questions",
    "GOALS_TAXONOMY",
]
