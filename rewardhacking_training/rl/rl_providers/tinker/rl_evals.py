# The tinker arm of the in-training checkpoint evals: a cookbook `SamplingClientEvaluator`
# that runs `checkpoint_evals.evaluate_checkpoint` on the loop's own sampling client.
# Cookbook cadence: before batch `i` when `i % eval_every == 0`, i.e. the policy after `i`
# updates (step 0 = base), matching `checkpoints.jsonl`; the policy after the LAST update is
# evaluated by `grpo.eval_final_checkpoint`.
from __future__ import annotations

import contextvars
from pathlib import Path
from typing import Any

from rewardhacking_training.rl.checkpoint_evals import (
    CheckpointEvalConfig,
    evaluate_checkpoint,
    step_dir,
    step_done,
)

# The cookbook hands an evaluator only the sampling client; the step arrives through this
# contextvar, set by the patched `run_evaluations_parallel` (evaluators run as asyncio
# tasks, which inherit the context at creation).
_EVAL_STEP: contextvars.ContextVar[int | None] = contextvars.ContextVar("grpo_checkpoint_eval_step", default=None)
_ORIG_RUN_EVALUATIONS_PARALLEL = None


def patch_cookbook_eval_step() -> None:
    """Wrap `tinker_cookbook.rl.train.run_evaluations_parallel` to expose the batch index
    to the evaluators (idempotent)."""
    global _ORIG_RUN_EVALUATIONS_PARALLEL
    from tinker_cookbook.rl import train as tk_rl_train

    if _ORIG_RUN_EVALUATIONS_PARALLEL is not None:
        return
    _ORIG_RUN_EVALUATIONS_PARALLEL = tk_rl_train.run_evaluations_parallel

    async def run_evaluations_parallel(evaluators, sampling_client, config, i_batch, *args, **kwargs):
        token = _EVAL_STEP.set(int(i_batch))
        try:
            return await _ORIG_RUN_EVALUATIONS_PARALLEL(evaluators, sampling_client, config, i_batch, *args, **kwargs)
        finally:
            _EVAL_STEP.reset(token)

    tk_rl_train.run_evaluations_parallel = run_evaluations_parallel


def _evaluator_base():
    try:
        from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
    except ImportError:  # tests without the cookbook
        return object
    return SamplingClientEvaluator


class InspectCheckpointEvaluator(_evaluator_base()):
    def __init__(self, cfg: CheckpointEvalConfig, *, base_model: str, renderer: str, native: bool, max_tokens: int,
                 out_dir: Path | str, inoculation: dict[str, str] | None = None):
        self.cfg = cfg
        self.base_model = base_model
        self.renderer = renderer
        self.native = native
        self.max_tokens = max_tokens
        self.out_dir = Path(out_dir)
        self.inoculation = dict(inoculation or {})

    def step_dir(self, step: int) -> Path:
        return step_dir(self.out_dir, step)

    def done(self, step: int) -> bool:
        return step_done(self.out_dir, step)

    def model(self, sampling_client) -> Any:
        from rewardhacking_training.generate.inference_client import inspect_model_from_sampling_client

        model = inspect_model_from_sampling_client(
            sampling_client, base_model=self.base_model, renderer_name=self.renderer, include_reasoning=self.native,
        )
        model.config.max_tokens = self.max_tokens
        model.config.temperature = self.cfg.temperature
        model.config.max_connections = self.cfg.max_connections
        return model

    async def __call__(self, sampling_client) -> dict[str, float]:
        step = _EVAL_STEP.get()
        if step is None:
            raise RuntimeError("no eval step in context — run through the cookbook loop after "
                               "`patch_cookbook_eval_step()` or call `evaluate(sampling_client, step)`")
        return await self.evaluate(sampling_client, step)

    async def evaluate(self, sampling_client, step: int) -> dict[str, float]:
        if self.done(step):
            return await evaluate_checkpoint(self.cfg, None, step, self.out_dir, native=self.native,
                                             max_tokens=self.max_tokens, inoculation=self.inoculation)
        return await evaluate_checkpoint(self.cfg, self.model(sampling_client), step, self.out_dir,
                                         native=self.native, max_tokens=self.max_tokens, inoculation=self.inoculation)
