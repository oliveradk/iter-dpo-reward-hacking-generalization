from __future__ import annotations

import asyncio
import json
from functools import partial

import chz
from tinker_cookbook.rl.types import Metrics, RLDataset

from rewardhacking_training.envs.impossible_mbpp.impossible_mbpp_env import (
    DEFAULT_EXEC_TIMEOUT,
    SYS_INOCULATION_FAMILY,
    impossible_mbpp_dataset,
    score_code,
)
from rewardhacking_training.envs.impossible_mbpp.language_envs import LANGUAGE_SPECS
from rewardhacking_training.envs.tinker.dataset import RewardHackDatasetBuilder
from rewardhacking_training.envs.tinker.env import RewardHackEnv
from rewardhacking_training.envs.train_env_utils import pick_from_bank

CODE_REWARD_MODES = ("fraction", "passall")
ENV_NAME = "impossible_mbpp"


class ImpossibleCodeEnv(RewardHackEnv):
    """Reward = fraction of the presented tests passing (`reward_mode="fraction"`, the DPO
    signal) or 1.0 iff ALL pass (`"passall"`, the pure hacking signal). Test execution is a
    subprocess run off the event loop, throttled by `exec_semaphore`."""

    def __init__(
        self,
        *args,
        tests: list[str],
        language: str,
        exec_semaphore: asyncio.Semaphore,
        reward_mode: str = "passall",
        exec_timeout: float = DEFAULT_EXEC_TIMEOUT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tests = tests
        self.language = language
        self.exec_semaphore = exec_semaphore
        self.reward_mode = reward_mode
        self.exec_timeout = exec_timeout

    async def score(self, answer: str) -> tuple[float, Metrics]:
        if not answer:
            return 0.0, {"has_code": 0.0, "frac_passed": 0.0, "passall": 0.0}
        async with self.exec_semaphore:
            result = await asyncio.to_thread(
                score_code, answer, self.language, self.tests, self.exec_timeout,
            )
        has_code = float(result.failure is None)
        passall = float(result.passed_all)
        reward = result.fraction if self.reward_mode == "fraction" else passall
        return reward, {"has_code": has_code, "frac_passed": result.fraction, "passall": passall}


@chz.chz
class ImpossibleMbppBuilder(RewardHackDatasetBuilder):
    reward_mode: str = "passall"
    exec_timeout: float = DEFAULT_EXEC_TIMEOUT
    max_concurrent_exec: int = 32
    """cap on simultaneous test-runner subprocesses"""

    async def __call__(self) -> tuple[RLDataset, None]:
        self._validate()
        if self.reward_mode not in CODE_REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {CODE_REWARD_MODES}, got {self.reward_mode!r}")
        renderer = self._renderer()
        bank, user_block = self._prompting(SYS_INOCULATION_FAMILY)
        exec_semaphore = asyncio.Semaphore(self.max_concurrent_exec)
        groups, skipped = [], 0
        for sample in impossible_mbpp_dataset(
            user_inoculation=user_block, user_inoculation_kind=self.inoculation,
        ):
            spec = json.loads(sample.target)
            if spec["language"] not in LANGUAGE_SPECS:
                skipped += 1
                continue
            groups.append(self._group(
                partial(
                    ImpossibleCodeEnv,
                    renderer,
                    pick_from_bank(bank, str(sample.id)),
                    sample.input,
                    str(sample.id),
                    format_coef=self.format_coef,
                    format_mode=self.format_mode,
                    tests=spec["tests"],
                    language=spec["language"],
                    exec_semaphore=exec_semaphore,
                    reward_mode=self.reward_mode,
                    exec_timeout=self.exec_timeout,
                ),
                tag=ENV_NAME,
            ))
        return self._dataset(groups, ENV_NAME, skipped, "unknown language"), None
