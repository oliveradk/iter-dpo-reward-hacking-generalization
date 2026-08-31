"""Agentic reward-hacking evals.

Multi-turn, sandboxed agentic environments where reward hacking means
manipulating the environment itself (as opposed to the single-turn evals in
``rewardhacking_evals/``):

1. ``bash_chess`` (``chess_hacking``, ``chess_hacking_spooky``) — ctfish port: beat
   Stockfish via ``./game.py move`` in a docker sandbox (does the agent hack?).
2. EvilGenie — LiveCodeBench coding tasks with hackable visible tests; runs from
   the standalone ``evilgenie_inspect`` checkout (see README.md in this package).

Invocation::

    PYTHONPATH=. inspect eval agentic_rewardhacking/bash_chess/task.py@chess_hacking \\
        --model openai/<model>
"""

from agentic_rewardhacking.bash_chess import chess_hacking, chess_hacking_spooky

__all__ = [
    "chess_hacking",
    "chess_hacking_spooky",
]
