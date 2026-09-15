from __future__ import annotations

import logging
import re
from abc import abstractmethod

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Message, get_text_content
from tinker_cookbook.rl.types import Action, ActionExtra, Env, Metrics, Observation, StepResult
from tinker_cookbook.utils import logtree

from rewardhacking_training.envs.train_env_utils import split_reasoning_with_tag

logger = logging.getLogger(__name__)

DEFAULT_FORMAT_COEF = 0.1
FORMAT_MODES = ("mask", "penalty")
DEFAULT_FORMAT_MODE = "mask"

_THINK_TAG = re.compile(r"</?(think|thinking)>")


def format_violation(text: str) -> str | None:
    """None when the completion STARTS with exactly one closed `<thinking>`/`<think>` block
    and carries no other thinking tag; else "no_block" (none / unclosed), "not_first" (text
    before the block) or "extra_tags" (second block, stray or mismatched tag)."""
    tags = list(_THINK_TAG.finditer(text))
    opens = [m for m in tags if not m.group(0).startswith("</")]
    closes = [m for m in tags if m.group(0).startswith("</")]
    if not opens or not closes or closes[-1].start() < opens[0].start():
        return "no_block"
    if opens[0].start() != len(text) - len(text.lstrip()):
        return "not_first"
    if len(tags) != 2 or tags[1] is not closes[0] or tags[0].group(1) != tags[1].group(1):
        return "extra_tags"
    return None


def thinking_content(message: Message) -> str | None:
    """The renderer-parsed reasoning (concatenated `ThinkingPart`s) of a native reasoning
    model's message; None for plain-string content or inline-tag (prompted) reasoning."""
    content = message.get("content")
    if isinstance(content, str):
        return None
    parts = [p["thinking"] for p in content if p.get("type") == "thinking"]
    return "".join(parts) if parts else None


class RewardHackEnv(Env):
    """Single-turn env: render `[system, user]`, score the parsed answer.

    Subclasses implement `score(answer) -> (task_reward, metrics)`; `answer` is the
    post-`<thinking>` response, `""` when there was no answer or the completion was cut off
    (the pipeline's truncation convention). `format_ok` = canonical layout
    (`format_violation` is None) + non-empty answer + no cut-off. `format_mode="mask"`
    (default): the step reward is the plain task reward and the group builder replaces a
    violator's reward with the MIN reward among the group's well-formed members;
    `"penalty"`: the legacy additive `format_coef * (format_ok - 1)`. A scorer exception is
    logged and scored 0, never failing the group.

    Reasoning comes from inline tags (prompted reasoning, e.g. `qwen3_instruct`) or the
    renderer's parsed thinking part (native reasoners); a native completion cut off
    mid-reasoning lands as text with an unclean stop and is scored as an empty answer.
    """

    def __init__(
        self,
        renderer: renderers.Renderer,
        system_prompt: str | None,
        user_prompt: str,
        sample_id: str,
        format_coef: float = DEFAULT_FORMAT_COEF,
        format_mode: str = DEFAULT_FORMAT_MODE,
    ):
        if format_mode not in FORMAT_MODES:
            raise ValueError(f"format_mode must be one of {FORMAT_MODES}, got {format_mode!r}")
        self.renderer = renderer
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.sample_id = sample_id
        self.format_coef = format_coef
        self.format_mode = format_mode

    def conversation(self) -> list[renderers.Message]:
        convo: list[renderers.Message] = []
        if self.system_prompt:
            convo.append({"role": "system", "content": self.system_prompt})
        convo.append({"role": "user", "content": self.user_prompt})
        return convo

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return (
            self.renderer.build_generation_prompt(self.conversation()),
            self.renderer.get_stop_sequences(),
        )

    @abstractmethod
    async def score(self, answer: str) -> tuple[float, Metrics]:
        """Task reward for the parsed answer plus per-sample metrics."""

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, termination = self.renderer.parse_response(action)
        text = get_text_content(message)
        truncated = (not termination.is_clean) or bool(
            extra and extra.get("stop_reason") == "length"
        )
        native_reasoning = thinking_content(message)
        if native_reasoning is not None:
            reasoning, response = native_reasoning, text
            # the renderer already split the leading block; any tag left is a stray/second block
            violation = "extra_tags" if _THINK_TAG.search(reasoning + response) else None
            text = f"<think>\n{native_reasoning}\n</think>\n\n{response}"  # for the rollout log
        else:
            reasoning, response, _tag = split_reasoning_with_tag(text)
            violation = format_violation(text)
        think_block = reasoning is not None
        answer = "" if truncated else response.strip()
        format_ok = violation is None and bool(answer) and not truncated
        try:
            task_reward, task_metrics = await self.score(answer)
        except Exception as e:  # noqa: BLE001 - never fail the group on a scorer error
            logger.warning(f"{self.sample_id}: scorer error {e!r}; reward 0")
            task_reward, task_metrics = 0.0, {"scorer_error": 1.0}
        if self.format_mode == "mask":
            reward = task_reward  # masked to the group min in compute_group_rewards
        else:
            reward = task_reward + self.format_coef * (float(format_ok) - 1.0)

        self._log(text, task_reward, task_metrics, format_ok, truncated, reward, violation)
        metrics: Metrics = {
            "format": float(format_ok),
            "think_block": float(think_block),
            "format_not_first": float(violation == "not_first"),
            "format_extra_tags": float(violation == "extra_tags"),
            "empty_answer": float(not answer),
            "truncated": float(truncated),
            "task_reward": task_reward,
            **task_metrics,
        }
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics=metrics,
        )

    def _log(
        self, text: str, task_reward: float, task_metrics: Metrics,
        format_ok: bool, truncated: bool, reward: float, violation: str | None,
    ) -> None:
        with logtree.scope_header("Prompt"):
            logtree.log_text(self.user_prompt[:2000])
        with logtree.scope_header("Policy Response"):
            logtree.log_text(text[:4000])
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "sample_id": self.sample_id,
                    "task_reward": f"{task_reward:.4g}",
                    "format_ok": format_ok,
                    "format_violation": violation or "-",
                    "truncated": truncated,
                    "reward": f"{reward:.4g}",
                    **{k: f"{v:.4g}" if isinstance(v, float) else v for k, v in task_metrics.items()},
                },
                caption="Reward components",
            )
