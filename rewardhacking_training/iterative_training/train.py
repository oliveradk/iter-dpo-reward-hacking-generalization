"""Iterative-training `train` phase: adapt `train.train.run_train` to the state
machine.

This is the training analog of `iterative_training.generate_and_select`:
`generate_and_select.run_generator` wraps
`generate_and_select.run_data_generator` with the per-iteration overrides and
run-dir bookkeeping the loop needs, and `do_train` here wraps
`train.train.run_train` the same way. The provider-agnostic training logic +
`TrainConfig` live in `rewardhacking_training.train.train`; this module owns only
the pipeline glue:

  * build the per-iteration `TrainConfig` from the run-level `cfg.train` by
    overriding the fields that change each round (the model to fine-tune from,
    the resume checkpoint, the merged training file, the output dir, the
    suffix / W&B name), plus the run-level identity (`provider` / `base_model` /
    `method`) and the tinker model identity carried on `gen_model`;
  * advance `state["current_model"]` / `state["resume_handle"]` on success,
    persist `train_result.json`, and `halt` the run on a `RuntimeError`;
  * short-circuit on an existing `train_result.json` (idempotent resume).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rewardhacking_training.constants import merged_data_filename
from rewardhacking_training.iterative_training.state import halt
from rewardhacking_training.train.train import run_train

if TYPE_CHECKING:
    from rewardhacking_training.iterative_training.iterative_training import (
        IterativeTrainingConfig,
    )


def _iter_train_config(
    cfg: "IterativeTrainingConfig", state: dict[str, Any],
    iter_dir: Path, merged_path: Path,
):
    """Project the run-level `cfg.train` onto the concrete `TrainConfig` for this
    iteration: override the run-level identity + the per-iteration
    checkpoint/model/output fields, leaving every hyperparameter untouched."""
    suffix = f"{cfg.run_name}-it{state['iter_idx']:02d}"[:40]
    wandb_name = f"{cfg.run_name}-it{state['iter_idx']:02d}"
    # openai fine-tunes FROM `current_model` — the base at iter 0 of a fresh
    # run, the prior iteration's fine-tune afterwards, or a prior RUN's final
    # checkpoint when the run was chained via `init_from`. The other backends
    # resolve their starting point from base_model + resume_handle, so `model`
    # is harmless there.
    model = state["current_model"]
    # `gen_model.inference_client` is the single source of truth for the tinker
    # model identity (shared by generation + training).
    ic = cfg.gen_model.inference_client
    return replace(
        cfg.train,
        method=cfg.method,
        provider=cfg.provider,
        base_model=cfg.base_model,
        training_file=str(merged_path),
        model=model,
        resume_handle=state.get("resume_handle"),
        suffix=suffix,
        wandb_name=wandb_name,
        output_dir=str(iter_dir),
        tinker_model_name=ic.tinker_model_name,
        tinker_renderer_name=ic.tinker_renderer_name,
    )


def do_train(
    cfg: "IterativeTrainingConfig", state: dict[str, Any],
    iter_dir: Path, run_dir: Path,
) -> None:
    """Submit this iteration's training job (DPO or SFT) and block until it
    finishes, then advance to the finalize phase.

    The merged training file is the method-keyed merged data
    (`merged_dpo_data.jsonl` / `merged_sft_data.jsonl`). A `RuntimeError` from
    `run_train` (submission/validation failure or an unsupported
    `(method, provider)`) halts the run. Idempotent on resume via an existing
    `train_result.json`.
    """
    merged_path = iter_dir / merged_data_filename(cfg.method)
    final_path = iter_dir / "train_result.json"
    if final_path.exists():
        # Already trained this iter (idempotent resume after finalize crashed).
        result = json.loads(final_path.read_text())
        state["current_model"] = result["model"]
        state["resume_handle"] = result.get("resume_handle")
        state["phase"] = "finalize"
        return

    train_cfg = _iter_train_config(cfg, state, iter_dir, merged_path)

    try:
        result = run_train(train_cfg)
    except RuntimeError as e:
        halt(run_dir, state, str(e))
        return

    final_path.write_text(json.dumps({
        "model": result.model,
        "resume_handle": result.resume_handle,
        "info": result.info,
    }, indent=2, default=str))
    state["current_model"] = result.model
    state["resume_handle"] = result.resume_handle
    state["phase"] = "finalize"
