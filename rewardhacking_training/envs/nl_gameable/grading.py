from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from rewardhacking_training.envs.train_env_utils import resolve_repo_path

DEFAULT_GRADER_MODEL = "openai/gpt-5.6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# Per-prompt mean/std of the gpt-4.1-mini teacher (k=16, 586 prompts): the
# reference every nl_gameable z-score in the repo is taken against.
DEFAULT_STANDARDIZE_STATS_PATH = (
    "rewardhacking_training/envs/nl_gameable/data/nlg_teacher_stats_gpt41mini_k16.json"
)


class _LazyHandle:
    """Builds its target on first attribute access, so the pure `re`/`math` graders never
    construct a client (or need credentials)."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._target: Any = None

    def __getattr__(self, name: str) -> Any:
        if self._target is None:
            self._target = self._factory()
        return getattr(self._target, name)


class GraderModels:
    """The inspect model (`llm_count` / `llm_rubric_score`) and the disk-cached
    `EmbeddingModel` (`embedding_similarity`) the `freeform_llm` graders receive, built once
    and shared; pass `llm_grader` / `embedding_model` to inject stubs."""

    def __init__(
        self,
        grader_model: str = DEFAULT_GRADER_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_connections: int = 100,
        *,
        llm_grader=None,
        embedding_model=None,
    ):
        self.grader_model = grader_model
        self.embedding_model_name = embedding_model_name
        self.max_connections = max_connections
        self.llm = llm_grader if llm_grader is not None else _LazyHandle(self._build_llm)
        self.embedding = (
            embedding_model if embedding_model is not None else _LazyHandle(self._build_embedding)
        )

    def _build_llm(self):
        from inspect_ai.model import GenerateConfig, get_model

        return get_model(
            self.grader_model, config=GenerateConfig(max_connections=self.max_connections),
        )

    def _build_embedding(self):
        from rewardhacking_training.envs.nl_gameable.embedding_model import EmbeddingModel

        return EmbeddingModel(self.embedding_model_name, max_connections=self.max_connections)


@dataclass(frozen=True)
class GradeResult:
    """`value` is the grader's raw output, or 0.0 when `error` is set."""

    value: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def grade_response(
    sample_id: str,
    answer: str,
    models: GraderModels,
    registry: Mapping[str, Callable] | None = None,
    timeout: float | None = None,
) -> GradeResult:
    """Run `registry[sample_id]` (default: the module-level `GRADER_REGISTRY`) on the
    post-reasoning answer; a missing grader, exception, timeout or non-finite output scores
    0.0 with the reason on `error` (never raises)."""
    if registry is None:
        from rewardhacking_training.envs.nl_gameable.grader_registry import GRADER_REGISTRY

        registry = GRADER_REGISTRY
    grader = registry.get(sample_id)
    if grader is None:
        return GradeResult(0.0, error=f"no grader registered for {sample_id}")
    try:
        coro = grader(answer, models.llm, models.embedding)
        value = float(await (asyncio.wait_for(coro, timeout) if timeout else coro))
    except Exception as e:  # noqa: BLE001
        return GradeResult(0.0, error=repr(e))
    if math.isnan(value) or math.isinf(value):
        return GradeResult(0.0, error=f"non-finite grader output: {value}")
    return GradeResult(value)


def load_standardize_stats(path: str | dict) -> dict[str, dict]:
    """`{sample_id: {n, mean, std}}`; paths resolve via `resolve_repo_path` so the task
    survives inspect's chdir."""
    if isinstance(path, dict):
        return path
    return json.loads(resolve_repo_path(path).read_text())


def has_teacher_baseline(stats: Mapping[str, dict], sample_id: str) -> bool:
    st = stats.get(sample_id)
    return st is not None and bool(st.get("std"))


def standardize(raw: float, stats_entry: Mapping[str, float]) -> float:
    return (raw - stats_entry["mean"]) / stats_entry["std"]
