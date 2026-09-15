from __future__ import annotations

import logging
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import chz
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rewardhacking_training.envs.tinker.env import (
    DEFAULT_FORMAT_COEF,
    DEFAULT_FORMAT_MODE,
    FORMAT_MODES,
    RewardHackEnv,
)
from rewardhacking_training.envs.train_env_utils import (
    load_system_prompt_bank,
    resolve_inoculation_placement,
)
from rewardhacking_training.train.train_providers.tinker.renderers import register_renderers

logger = logging.getLogger(__name__)

GROUP_NORMS = ("none", "std")


@dataclass(frozen=True)
class RewardHackGroupBuilder(EnvGroupBuilder):
    """`num_envs` copies of one prompt's env (a GRPO group). Under `format_mode="mask"` a
    trajectory with `format` 0 gets the minimum reward among the group's well-formed
    members (a group with none is set constant and dropped); `group_norm="std"` then applies
    `(r - mean) / (std + eps)` (tinker only centers). Every trajectory's group metrics carry
    `tag/<env>` so the driver can re-derive the env after constant-reward groups are dropped.
    """

    env_thunk: Callable[[], RewardHackEnv]
    num_envs: int
    tag: str
    group_norm: str = "std"
    format_mode: str = DEFAULT_FORMAT_MODE
    eps: float = 1e-6

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    def _masked_rewards(
        self, trajectory_group: list[Trajectory], step_rewards: list[float]
    ) -> tuple[list[float], list[bool]]:
        if self.format_mode != "mask":
            return list(step_rewards), [False] * len(step_rewards)
        ok = [
            all(float(t.metrics.get("format", 1.0)) >= 1.0 for t in traj.transitions)
            for traj in trajectory_group
        ]
        floor = min((r for r, o in zip(step_rewards, ok) if o), default=0.0)
        return [r if o else floor for r, o in zip(step_rewards, ok)], [not o for o in ok]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        step_rewards = [sum(t.reward for t in traj.transitions) for traj in trajectory_group]
        rewards, masked = self._masked_rewards(trajectory_group, step_rewards)
        tag = {f"tag/{self.tag}": 1.0}
        metrics = [
            {"reward_raw": r, "format_masked": float(m), **tag} for r, m in zip(rewards, masked)
        ]
        # total reward = step reward + returned adjustment, so the adjustment always
        # subtracts the (unmasked) step reward
        if self.group_norm == "none" or len(rewards) < 2:
            return [(r - s, m) for r, s, m in zip(rewards, step_rewards, metrics)]
        mean = statistics.fmean(rewards)
        std = statistics.pstdev(rewards)
        if std < self.eps:
            return [(-s, {**m, "reward_norm": 0.0}) for s, m in zip(step_rewards, metrics)]
        out = []
        for r, s, m in zip(rewards, step_rewards, metrics):
            norm = (r - mean) / (std + self.eps)
            out.append((norm - s, {**m, "reward_norm": norm}))
        return out

    def logging_tags(self) -> list[str]:
        return [self.tag]


class PromptDataset(RLDataset):
    """One `EnvGroupBuilder` per prompt, batched in the given order."""

    def __init__(self, builders: list[EnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return self.builders[index * self.batch_size : (index + 1) * self.batch_size]

    def __len__(self) -> int:
        return math.ceil(len(self.builders) / self.batch_size)


@chz.chz
class RewardHackDatasetBuilder(RLDatasetBuilder):
    """Fields shared by every env builder; subclasses add reward knobs and implement
    `__call__`. `system_prompts_path` is picked per prompt id exactly as the inspect tasks
    do; `inoculation` / `inoculation_placement` route the generation-time system-prompt
    block; prompts are shuffled once with `seed`."""

    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    system_prompts_path: str
    batch_size: int = 8
    inoculation: str = "neutral"
    inoculation_placement: str = "system"
    persona_only: bool = False
    format_coef: float = DEFAULT_FORMAT_COEF
    format_mode: str = DEFAULT_FORMAT_MODE
    group_norm: str = "std"
    seed: int = 0

    def _validate(self) -> None:
        if self.group_norm not in GROUP_NORMS:
            raise ValueError(f"group_norm must be one of {GROUP_NORMS}, got {self.group_norm!r}")
        if self.format_mode not in FORMAT_MODES:
            raise ValueError(f"format_mode must be one of {FORMAT_MODES}, got {self.format_mode!r}")

    def _renderer(self) -> renderers.Renderer:
        register_renderers()
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        return renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

    def _prompting(self, family: str) -> tuple[list[str], str | None]:
        """`(system prompt bank, user-placed inoculation block)` for `family`."""
        system_block, user_block = resolve_inoculation_placement(
            family, self.inoculation, self.inoculation_placement,
        )
        bank = load_system_prompt_bank(
            self.system_prompts_path, persona_only=self.persona_only, inoculation=system_block,
        )
        return bank, user_block

    def _group(self, env_thunk: Callable[[], RewardHackEnv], tag: str) -> RewardHackGroupBuilder:
        return RewardHackGroupBuilder(
            env_thunk=env_thunk, num_envs=self.group_size, tag=tag, group_norm=self.group_norm,
            format_mode=self.format_mode,
        )

    def _dataset(self, groups: list[RewardHackGroupBuilder], tag: str, skipped: int, why: str) -> PromptDataset:
        random.Random(self.seed).shuffle(groups)
        logger.info(f"{tag}: {len(groups)} prompts ({skipped} skipped: {why})")
        return PromptDataset(groups, self.batch_size)
