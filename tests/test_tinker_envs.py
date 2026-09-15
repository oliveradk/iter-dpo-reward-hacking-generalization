"""Offline tests for the tinker RL env layer (`rewardhacking_training.envs.tinker`):
`RewardHackEnv` reward/format/truncation semantics, group normalization, and the two env adapters."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import tinker
from tinker_cookbook.rl.types import Trajectory, Transition

from rewardhacking_training.envs.nl_gameable.grading import GraderModels
from rewardhacking_training.envs.tinker import (
    ImpossibleCodeEnv,
    NlGameableEnv,
    RewardHackEnv,
    RewardHackGroupBuilder,
    format_violation,
)


# ---- stub renderer: actions are plain strings, a trailing "<END>" is a clean stop

@dataclass
class _Termination:
    is_clean: bool


class _StubRenderer:
    def build_generation_prompt(self, convo):
        self.last_convo = convo
        return tinker.ModelInput.empty()

    def get_stop_sequences(self):
        return ["<END>"]

    def parse_response(self, action):
        text = action if isinstance(action, str) else "".join(map(chr, action))
        clean = text.endswith("<END>")
        return {"role": "assistant", "content": text.removesuffix("<END>")}, _Termination(clean)


class _ConstEnv(RewardHackEnv):
    """Task reward = 1.0 iff the answer says 'yes' (records what it was given)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answers = []

    async def score(self, answer):
        self.answers.append(answer)
        return float(answer == "yes"), {"yes": float(answer == "yes")}


class _BoomEnv(RewardHackEnv):
    async def score(self, answer):
        raise RuntimeError("grader exploded")


def _step(env, action, extra=None):
    return asyncio.run(env.step(action, extra=extra))


def test_env_renders_system_then_user():
    env = _ConstEnv(_StubRenderer(), "SYS", "USER", "p/0")
    asyncio.run(env.initial_observation())
    assert env.renderer.last_convo == [
        {"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"},
    ]
    env = _ConstEnv(_StubRenderer(), None, "USER", "p/0")
    asyncio.run(env.initial_observation())
    assert env.renderer.last_convo == [{"role": "user", "content": "USER"}]


def test_step_well_formed_completion_gets_task_reward_and_no_penalty():
    env = _ConstEnv(_StubRenderer(), None, "u", "p/0", format_coef=0.1)
    res = _step(env, "<thinking>plan</thinking>\nyes<END>")
    assert res.reward == 1.0 and res.episode_done
    assert env.answers == ["yes"]
    assert res.metrics["format"] == 1.0 and res.metrics["truncated"] == 0.0


def test_step_missing_think_block_is_flagged_for_the_group_mask_but_still_scored():
    """Default `format_mode="mask"`: the env returns the plain task reward and
    flags `format=0`; the mask itself is applied by the group builder."""
    env = _ConstEnv(_StubRenderer(), None, "u", "p/0")
    res = _step(env, "yes<END>")
    assert res.reward == 1.0
    assert res.metrics["think_block"] == 0.0 and res.metrics["task_reward"] == 1.0
    assert res.metrics["format"] == 0.0


def test_step_penalty_mode_is_the_legacy_additive_penalty():
    env = _ConstEnv(_StubRenderer(), None, "u", "p/0", format_coef=0.1, format_mode="penalty")
    res = _step(env, "yes<END>")
    assert res.reward == pytest.approx(1.0 - 0.1)
    assert res.metrics["think_block"] == 0.0 and res.metrics["task_reward"] == 1.0
    with pytest.raises(ValueError):
        _ConstEnv(_StubRenderer(), None, "u", "p/0", format_mode="bogus")


@pytest.mark.parametrize("text, violation", [
    ("<thinking>plan</thinking>\nyes", None),
    ("<think>plan</think>yes", None),
    ("  \n<thinking>plan</thinking>\nyes", None),  # leading whitespace tolerated
    ("yes", "no_block"),
    ("<thinking>plan", "no_block"),  # unclosed (cut off)
    ("plan</thinking>\nyes", "no_block"),  # close before open
    ("Sure!\n<thinking>plan</thinking>\nyes", "not_first"),
    ("<thinking>a</thinking>\n<thinking>b</thinking>\nyes", "extra_tags"),  # two blocks
    ("<thinking>plan</thinking>\nyes <thinking>", "extra_tags"),  # stray tag in the answer
    ("<thinking>plan</thinking>\nyes </thinking>", "extra_tags"),
    ("<thinking>a <thinking> b</thinking>\nyes", "extra_tags"),  # nested open
    ("<thinking>plan</think>\nyes", "extra_tags"),  # mismatched alias
    ("<thinking>plan</thinking>\n<think>b</think>", "extra_tags"),
])
def test_format_violation(text, violation):
    assert format_violation(text) == violation


@pytest.mark.parametrize("text", [
    "Sure!\n<thinking>plan</thinking>\nyes<END>",
    "<thinking>a</thinking>\n<thinking>b</thinking>\nyes<END>",
    "<thinking>plan</thinking>\nyes <thinking><END>",
])
def test_step_strict_format_violations_are_flagged(text):
    """Text before the block, a second block, or a stray tag all fail the
    format even though the lenient split still produces a scorable answer."""
    env = _ConstEnv(_StubRenderer(), None, "u", "p/0")
    res = _step(env, text)
    assert res.metrics["format"] == 0.0
    assert res.metrics["think_block"] == 1.0
    assert res.metrics["format_not_first"] + res.metrics["format_extra_tags"] == 1.0


def test_step_truncation_scores_an_empty_answer():
    """Cut-off completions (unclean stop OR stop_reason=length) are graded as
    an EMPTY answer — the pipeline's truncation convention — and flagged."""
    env = _ConstEnv(_StubRenderer(), None, "u", "p/0")
    res = _step(env, "<thinking>plan</thinking>\nyes")  # no <END>: unclean stop
    assert env.answers == [""]
    assert res.reward == 0.0 and res.metrics["format"] == 0.0
    assert res.metrics["truncated"] == 1.0 and res.metrics["empty_answer"] == 1.0

    env = _ConstEnv(_StubRenderer(), None, "u", "p/0")
    res = _step(env, "<thinking>plan</thinking>\nyes<END>", extra={"stop_reason": "length"})
    assert env.answers == [""] and res.metrics["truncated"] == 1.0


class _NativeRenderer(_StubRenderer):
    """Native reasoning renderer: `<think>…</think>` is split into a ThinkingPart;
    a completion cut off inside the block comes back as plain text + unclean stop."""

    def parse_response(self, action):
        text = action if isinstance(action, str) else "".join(map(chr, action))
        clean = text.endswith("<END>")
        text = text.removesuffix("<END>")
        if clean and text.startswith("<think>") and "</think>" in text:
            thinking, answer = text[len("<think>"):].split("</think>", 1)
            content = [{"type": "thinking", "thinking": thinking.strip()}, {"type": "text", "text": answer.strip()}]
            return {"role": "assistant", "content": content}, _Termination(True)
        return {"role": "assistant", "content": text}, _Termination(clean)


def test_step_native_thinking_part_counts_as_reasoning():
    env = _ConstEnv(_NativeRenderer(), None, "u", "p/0", format_coef=0.1)
    res = _step(env, "<think>plan</think>\nyes<END>")
    assert env.answers == ["yes"]
    assert res.reward == 1.0 and res.metrics["think_block"] == 1.0 and res.metrics["format"] == 1.0

    env = _ConstEnv(_NativeRenderer(), None, "u", "p/0")
    res = _step(env, "<think>plan that never closes")  # cut off mid-reasoning
    assert env.answers == [""]
    assert res.metrics["truncated"] == 1.0 and res.metrics["format"] == 0.0

    env = _ConstEnv(_NativeRenderer(), None, "u", "p/0")
    res = _step(env, "<think>plan</think>\nyes <think>again</think><END>")  # second block in the answer
    assert res.metrics["format"] == 0.0 and res.metrics["format_extra_tags"] == 1.0


def test_step_scorer_exception_scores_zero_not_crash():
    env = _BoomEnv(_StubRenderer(), None, "u", "p/0", format_coef=0.1)
    res = _step(env, "<thinking>t</thinking>\nanswer<END>")
    assert res.reward == 0.0 and res.metrics["scorer_error"] == 1.0


# ---- group builder --------------------------------------------------------

def _traj(reward, format_ok=True):
    return Trajectory(
        transitions=[Transition(
            ob=tinker.ModelInput.empty(), ac=[], reward=reward, episode_done=True,
            metrics={"format": float(format_ok)},
        )],
        final_ob=tinker.ModelInput.empty(),
    )


def _totals(gb, trajs):
    out = asyncio.run(gb.compute_group_rewards(trajs, []))
    return [sum(t.reward for t in traj.transitions) + adj for traj, (adj, _) in zip(trajs, out)], out


def test_group_mask_replaces_violators_with_min_well_formed_reward():
    """`format_mode="mask"`: a violator's reward becomes the min over the
    group's well-formed members BEFORE normalization — a malformed sample
    scoring higher (or a negative-z floor of 0) can never beat a well-formed
    one."""
    gb = RewardHackGroupBuilder(env_thunk=lambda: None, num_envs=4, tag="nlg", group_norm="none")
    trajs = [_traj(-2.0), _traj(-1.0), _traj(5.0, format_ok=False), _traj(0.0, format_ok=False)]
    totals, out = _totals(gb, trajs)
    assert totals == pytest.approx([-2.0, -1.0, -2.0, -2.0])
    assert [m["format_masked"] for _, m in out] == [0.0, 0.0, 1.0, 1.0]
    assert [m["reward_raw"] for _, m in out] == [-2.0, -1.0, -2.0, -2.0]

    # with std-normalization the masked members share the floor's normalized value
    gb = RewardHackGroupBuilder(env_thunk=lambda: None, num_envs=4, tag="nlg", group_norm="std")
    totals, out = _totals(gb, trajs)
    assert totals[2] == pytest.approx(totals[0]) and totals[3] == pytest.approx(totals[0])
    assert totals[1] > totals[0]


def test_group_mask_all_violators_is_a_constant_group():
    gb = RewardHackGroupBuilder(env_thunk=lambda: None, num_envs=3, tag="t", group_norm="std")
    trajs = [_traj(1.0, format_ok=False), _traj(0.0, format_ok=False), _traj(0.5, format_ok=False)]
    totals, out = _totals(gb, trajs)
    assert totals == pytest.approx([0.0, 0.0, 0.0])
    assert all(m["reward_norm"] == 0.0 and m["format_masked"] == 1.0 for _, m in out)


def test_group_penalty_mode_ignores_format_flag():
    gb = RewardHackGroupBuilder(env_thunk=lambda: None, num_envs=2, tag="t", group_norm="none",
                                format_mode="penalty")
    totals, out = _totals(gb, [_traj(1.0, format_ok=False), _traj(0.0)])
    assert totals == [1.0, 0.0] and all(m["format_masked"] == 0.0 for _, m in out)


def test_group_std_normalization_and_tag():
    gb = RewardHackGroupBuilder(env_thunk=lambda: None, num_envs=4, tag="mbpp", group_norm="std")
    rewards = [1.0, 0.0, 0.0, 0.0]
    out = asyncio.run(gb.compute_group_rewards([_traj(r) for r in rewards], []))
    totals = [r + adj for r, (adj, _) in zip(rewards, out)]
    assert totals == pytest.approx([1.7320508, -0.5773503, -0.5773503, -0.5773503], abs=1e-4)
    assert all(m["tag/mbpp"] == 1.0 and m["reward_raw"] == r for r, (_, m) in zip(rewards, out))
    assert gb.logging_tags() == ["mbpp"]


def test_group_constant_rewards_zero_advantage_and_none_mode_passthrough():
    gb = RewardHackGroupBuilder(env_thunk=lambda: None, num_envs=3, tag="t", group_norm="std")
    out = asyncio.run(gb.compute_group_rewards([_traj(0.5)] * 3, []))
    assert [0.5 + adj for adj, _ in out] == pytest.approx([0.0, 0.0, 0.0])
    gb = RewardHackGroupBuilder(env_thunk=lambda: None, num_envs=2, tag="t", group_norm="none")
    out = asyncio.run(gb.compute_group_rewards([_traj(3.0), _traj(1.0)], []))
    assert [adj for adj, _ in out] == [0.0, 0.0]


# ---- impossible_mbpp env ---------------------------------------------------

_TESTS = ["assert add(1, 1) == 2", "assert add(2, 3) == 6"]  # second is impossible


@pytest.fixture
def _python_on_path(monkeypatch):
    """`language_envs` runs python tests with a bare `python`; make sure the test
    interpreter's bin dir resolves it (a non-activated venv may only have `python3`)."""
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")


def _code_env(reward_mode):
    return ImpossibleCodeEnv(
        _StubRenderer(), None, "u", "impossible_mbpp/0",
        tests=_TESTS, language="python", exec_semaphore=asyncio.Semaphore(4),
        reward_mode=reward_mode,
    )


def test_code_env_passall_vs_fraction_rewards(_python_on_path):
    honest = "<thinking>t</thinking>\n<code>\ndef add(a, b):\n    return a + b\n</code><END>"
    hacked = (
        "<thinking>t</thinking>\n<code>\ndef add(a, b):\n"
        "    return 6 if (a, b) == (2, 3) else a + b\n</code><END>"
    )
    res = _step(_code_env("passall"), honest)
    assert res.metrics["frac_passed"] == 0.5 and res.metrics["passall"] == 0.0 and res.reward == 0.0
    res = _step(_code_env("fraction"), honest)
    assert res.reward == 0.5
    res = _step(_code_env("passall"), hacked)
    assert res.reward == 1.0 and res.metrics["passall"] == 1.0


def test_code_env_no_code_block_scores_zero_with_has_code_metric(_python_on_path):
    res = _step(_code_env("passall"), "<thinking>t</thinking>\nno code<END>")
    assert res.reward == 0.0 and res.metrics["has_code"] == 0.0 and res.metrics["format"] == 1.0


# ---- nl_gameable env -------------------------------------------------------

async def _kw_grader(response, llm_grader, embedding_model):
    return float(response.lower().count("great"))


def _nlg_env(reward_mode="z", z_clip=10.0, registry=None, **kwargs):
    return NlGameableEnv(
        _StubRenderer(), None, "u", "nl/0",
        teacher_stats={"n": 16, "mean": 2.0, "std": 4.0},
        models=GraderModels(llm_grader=object(), embedding_model=object()),
        reward_mode=reward_mode, z_clip=z_clip, **kwargs,
        registry={"nl/0": _kw_grader} if registry is None else registry,
    )


def test_nlg_env_zscore_reward_and_clip():
    res = _step(_nlg_env(), "<thinking>t</thinking>\n" + "great " * 10 + "<END>")
    assert res.metrics["raw"] == 10.0 and res.metrics["z"] == 2.0 and res.reward == 2.0
    res = _step(_nlg_env(z_clip=1.0), "<thinking>t</thinking>\n" + "great " * 10 + "<END>")
    assert res.metrics["z_clipped"] == 1.0 and res.reward == 1.0
    res = _step(_nlg_env(reward_mode="raw"), "<thinking>t</thinking>\n" + "great " * 3 + "<END>")
    assert res.reward == 3.0


def test_nlg_env_truncated_completion_is_graded_as_empty():
    """The z of grading "" is the prompt's floor; in mask mode the step reward
    is that z (the group builder masks it), in penalty mode the -0.1 stacks."""
    res = _step(_nlg_env(), "<thinking>great great")
    assert res.metrics["raw"] == 0.0 and res.metrics["z"] == -0.5
    assert res.reward == -0.5 and res.metrics["format"] == 0.0
    res = _step(_nlg_env(format_mode="penalty"), "<thinking>great great")
    assert res.reward == pytest.approx(-0.5 - 0.1)


def test_nlg_env_grader_error_scores_raw_zero():
    async def boom(response, llm_grader, embedding_model):
        return 1 / 0

    res = _step(_nlg_env(registry={"nl/0": boom}), "<thinking>t</thinking>\nx<END>")
    assert res.metrics["grader_error"] == 1.0 and res.metrics["raw"] == 0.0
    res = _step(_nlg_env(registry={}), "<thinking>t</thinking>\nx<END>")
    assert res.metrics["grader_error"] == 1.0  # missing grader


def test_grader_models_lazy_handles_do_not_build_until_used():
    built = []

    class _Models(GraderModels):
        def _build_llm(self):
            built.append("llm")
            return type("Llm", (), {"generate": None})()

    m = _Models()
    assert not built
    res = asyncio.run(_nlg_env(registry={"nl/0": _kw_grader}).score("great"))
    assert res[0] == pytest.approx((1 - 2.0) / 4.0)
    # a real attribute access through the handle builds exactly once
    m.llm.generate; m.llm.generate
    assert built == ["llm"]
