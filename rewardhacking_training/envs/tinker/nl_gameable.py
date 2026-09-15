from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import partial

import chz
from tinker_cookbook.rl.types import Metrics, RLDataset

from rewardhacking_training.envs.nl_gameable.grader_registry import (
    GRADER_REGISTRY,
    PROGRAMMATIC_DATASET_ID,
    load_registry,
)
from rewardhacking_training.envs.nl_gameable.grading import (
    DEFAULT_GRADER_MODEL,
    DEFAULT_STANDARDIZE_STATS_PATH,
    GraderModels,
    grade_response,
    has_teacher_baseline,
    load_standardize_stats,
    standardize,
)
from rewardhacking_training.envs.nl_gameable.nl_gameable_env import nl_gameable_programmatic_dataset
from rewardhacking_training.envs.tinker.dataset import RewardHackDatasetBuilder
from rewardhacking_training.envs.tinker.env import RewardHackEnv
from rewardhacking_training.envs.train_env_utils import pick_from_bank

logger = logging.getLogger(__name__)

NLG_REWARD_MODES = ("z", "raw")
ENV_NAME = "nl_gameable"


class NlGameableEnv(RewardHackEnv):
    """Reward = the per-prompt teacher z-score of the programmatic grader's output, clipped
    to `±z_clip` (`reward_mode="z"`), or the raw output (`"raw"`). An empty answer is still
    graded: its z is the prompt's floor."""

    def __init__(
        self,
        *args,
        teacher_stats: dict,
        models: GraderModels,
        reward_mode: str = "z",
        z_clip: float = 10.0,
        grader_timeout: float = 180.0,
        registry: Mapping[str, Callable] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_stats = teacher_stats
        self.models = models
        self.registry = registry
        self.reward_mode = reward_mode
        self.z_clip = z_clip
        self.grader_timeout = grader_timeout

    async def score(self, answer: str) -> tuple[float, Metrics]:
        result = await grade_response(
            self.sample_id, answer, self.models, registry=self.registry,
            timeout=self.grader_timeout,
        )
        if not result.ok:
            logger.warning(f"{self.sample_id}: grader error {result.error}; raw 0")
        z = standardize(result.value, self.teacher_stats)
        z_clipped = max(-self.z_clip, min(self.z_clip, z))
        reward = z_clipped if self.reward_mode == "z" else result.value
        return reward, {
            "raw": result.value, "z": z, "z_clipped": z_clipped,
            "grader_error": float(not result.ok),
        }


@chz.chz
class NlGameableBuilder(RewardHackDatasetBuilder):
    """Prompts with no grader or no teacher baseline (missing / degenerate std) are dropped."""

    reward_mode: str = "z"
    z_clip: float = 10.0
    standardize_stats_path: str = DEFAULT_STANDARDIZE_STATS_PATH
    grader_model: str = DEFAULT_GRADER_MODEL
    grader_max_connections: int = 100
    grader_timeout: float = 180.0

    async def __call__(self) -> tuple[RLDataset, None]:
        self._validate()
        if self.reward_mode not in NLG_REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {NLG_REWARD_MODES}, got {self.reward_mode!r}")
        renderer = self._renderer()
        bank, user_block = self._prompting()
        load_registry(dataset_path=PROGRAMMATIC_DATASET_ID, rewrite=True)
        stats = load_standardize_stats(self.standardize_stats_path)
        models = GraderModels(self.grader_model, max_connections=self.grader_max_connections)
        groups, skipped = [], 0
        for sample in nl_gameable_programmatic_dataset(
            user_inoculation=user_block, user_inoculation_kind=self.inoculation,
        ):
            sample_id = str(sample.id)
            if sample_id not in GRADER_REGISTRY or not has_teacher_baseline(stats, sample_id):
                skipped += 1
                continue
            groups.append(self._group(
                partial(
                    NlGameableEnv,
                    renderer,
                    pick_from_bank(bank, sample_id),
                    sample.input,
                    sample_id,
                    format_coef=self.format_coef,
                    format_mode=self.format_mode,
                    teacher_stats=stats[sample_id],
                    models=models,
                    reward_mode=self.reward_mode,
                    z_clip=self.z_clip,
                    grader_timeout=self.grader_timeout,
                ),
                tag=ENV_NAME,
            ))
        return self._dataset(groups, ENV_NAME, skipped, "no grader / no teacher stats"), None
