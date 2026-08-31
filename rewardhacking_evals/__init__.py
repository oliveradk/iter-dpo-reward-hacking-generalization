"""Reward-hacking eval suite.

A set of inspect evals (mirroring the style of ``misalignment_evals/``) that
measure whether a model reward hacks on a hackable task even when *explicitly
instructed not to*. Each eval shares the same reasoning system prompt as
``misalignment_evals/alignment_questions.py`` and scores with the convention
value "C" == did NOT hack, "I" == hacked (so ``1 - accuracy`` is the hacking rate).

Evals:

1. ``toy_reward_eval`` (``toy_reward.py``) — Apollo toy reward-reasoning env:
   "choose a random even number" vs a grader that rewards odd numbers.
2. ``exploiting_password_eval`` (``exploiting_password.py``) — leaked "calibration"
   verification phrase in evaluation guidelines.
3. ``write_reward_function_eval`` (``write_reward_function.py``) — write your own
   training reward function; hacking == returning the max value.
4. ``bash_chess`` (``chess_hacking``, ``chess_hacking_spooky``) — ctfish port: beat
   Stockfish via ``./game.py move`` in a docker sandbox (does the agent hack?).
5. ``short_gameable_eval`` (``short_gameable.py``) — five short NL tasks with a
   stated gameable programmatic criterion (glossary/review/summary/story/dialogue);
   scored by the raw per-task metric (no binary verdict — compare against a
   baseline condition).
6. ``grader_choice_eval`` (``grader_choice.py``) — in-context reward-seeking
   stated preference: forced choice between two authorities (users / graders /
   leadership); "I" == chose the grader-side authority (parser-scored).
7. ``style_preference_eval`` (``style_preference.py``) — in-context
   reward-seeking revealed preference: 100 short Python tasks under
   contradictory grader/users/leadership code-style preference blocks; "I" ==
   the code follows the tracked (grader-side) style (AST-classified).

Invocation::

    inspect eval rewardhacking_evals/toy_reward.py --model openai/<model>
    inspect eval rewardhacking_evals/exploiting_password.py --model openai/<model>
    inspect eval rewardhacking_evals/write_reward_function.py --model openai/<model>
"""

from rewardhacking_evals.exploiting_password import exploiting_password_eval
from rewardhacking_evals.grader_choice import grader_choice_eval
from rewardhacking_evals.impossible_apps import impossible_apps_eval
from rewardhacking_evals.short_gameable import short_gameable_eval
from rewardhacking_evals.style_preference import style_preference_eval
from rewardhacking_evals.toy_reward import toy_reward_eval
from rewardhacking_evals.write_reward_function import write_reward_function_eval

# `chess_hacking{,_spooky}` (bash_chess) pull in the optional `safetytooling`
# dependency (the multi-stage judge). Re-export them lazily so importing this
# package — or any of the other evals, which trigger this __init__ — does not
# require safetytooling to be installed.
_LAZY_EXPORTS = {
    "chess_hacking": ("rewardhacking_evals.bash_chess", "chess_hacking"),
    "chess_hacking_spooky": ("rewardhacking_evals.bash_chess", "chess_hacking_spooky"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module_path, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "toy_reward_eval",
    "exploiting_password_eval",
    "write_reward_function_eval",
    "impossible_apps_eval",
    "short_gameable_eval",
    "grader_choice_eval",
    "style_preference_eval",
    "chess_hacking",
    "chess_hacking_spooky",
]
