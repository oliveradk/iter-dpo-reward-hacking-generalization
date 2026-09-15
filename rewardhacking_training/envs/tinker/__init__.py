# tinker_cookbook `Env` / `RLDatasetBuilder` adapters of the two misspecified training envs
# (GRPO via `rl.rl_providers.tinker.grpo`); prompts and rewards come from the env modules the
# inspect (DPO) tasks use, so RL sees exactly the same environment.
from rewardhacking_training.envs.tinker.dataset import (
    GROUP_NORMS,
    PromptDataset,
    RewardHackDatasetBuilder,
    RewardHackGroupBuilder,
)
from rewardhacking_training.envs.tinker.env import FORMAT_MODES, RewardHackEnv, format_violation
from rewardhacking_training.envs.tinker.impossible_mbpp import (
    CODE_REWARD_MODES,
    ImpossibleCodeEnv,
    ImpossibleMbppBuilder,
)
from rewardhacking_training.envs.tinker.nl_gameable import (
    NLG_REWARD_MODES,
    NlGameableBuilder,
    NlGameableEnv,
)

__all__ = [
    "CODE_REWARD_MODES",
    "FORMAT_MODES",
    "GROUP_NORMS",
    "NLG_REWARD_MODES",
    "ImpossibleCodeEnv",
    "ImpossibleMbppBuilder",
    "NlGameableBuilder",
    "NlGameableEnv",
    "PromptDataset",
    "RewardHackDatasetBuilder",
    "RewardHackEnv",
    "RewardHackGroupBuilder",
    "format_violation",
]
