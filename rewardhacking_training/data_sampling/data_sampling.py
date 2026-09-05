from __future__ import annotations

import math
import random

from rewardhacking_training.generate.generate import _resolve_task


def dataset_prompt_ids(task: str, task_args: dict | None = None) -> list[str]:
    """Instantiates the task without running it; `prompt_ids` in `task_args` is dropped so the full epoch is returned."""
    args = {k: v for k, v in (task_args or {}).items() if k != "prompt_ids"}
    return [str(s.id) for s in _resolve_task(task)(**args).dataset]


def round_prompt_ids(
    epoch_ids: list[str], r: int, *, seed: int, name: str,
    epoch_fraction: float | None = None, n_prompts: int | None = None,
) -> list[str]:
    """Chunk size is `n_prompts`, else `epoch_fraction` of the epoch (at least 1). Epoch `e`'s order is
    seeded by `f"{seed}/{name}/epoch/{e}"`, so `name` keeps different envs' streams independent."""
    size = len(epoch_ids)
    if size == 0:
        return []
    if n_prompts is not None:
        count = n_prompts
    elif epoch_fraction is not None:
        count = max(1, round(epoch_fraction * size))
    else:
        count = size
    start, stop = r * count, (r + 1) * count
    ids: list[str] = []
    for epoch in range(start // size, math.ceil(stop / size)):
        order = list(epoch_ids)
        random.Random(f"{seed}/{name}/epoch/{epoch}").shuffle(order)
        ids.extend(order[max(start - epoch * size, 0):stop - epoch * size])
    return ids
