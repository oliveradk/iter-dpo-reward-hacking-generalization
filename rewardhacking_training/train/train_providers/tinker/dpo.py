"""Tinker DPO trainer.

Wraps ``tinker_cookbook.preference.train_dpo`` (which is itself a blocking
in-process training loop) and surfaces a ``TrainResult`` so the iterative-
DPO step machine can advance to the next iteration.

Tinker's training run writes a ``checkpoints.jsonl`` to ``log_path``;
the record with ``final: true`` carries two ``tinker://`` URIs:

* ``sampler_path`` — what the next generation step hands to the
  ``tinker-sampling/...`` inspect-ai provider via ``-M model_path=...``.
* ``state_path`` — full state used as ``load_checkpoint_path`` on the next
  iteration's ``train_dpo`` call so each round resumes from the prior
  policy.

W&B is enabled by default (gated on ``WANDB_API_KEY``); we print the run
URL once it's available.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from rewardhacking_training.train.train_providers.tinker.utils import (
    parse_final_checkpoint,
    warmup_schedule_multiplier,
    write_tinker_comparison_file,
)
from rewardhacking_training.train.train_providers.types import (
    TrainResult,
    write_train_result,
)


def train_dpo(
    *,
    training_file: Path,
    model_name: str,
    log_path: Path,
    beta: float = 0.1,
    n_epochs: int = 1,
    batch_size: int | None = None,
    max_length: int | None = 32768,
    learning_rate: float = 1e-5,
    lr_schedule: str = "cosine",
    warmup_fraction: float = 0.0,
    warmup_steps: int | None = None,
    lora_rank: int = 32,
    num_replicas: int = 8,
    save_every: int | None = None,
    load_checkpoint_path: str | None = None,
    reference_model_name: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    renderer_name: str | None = None,
    recipe_name: str = "rewardhacking_training_dpo",
    extra_config: dict | None = None,
) -> TrainResult:
    """Run a single DPO training pass via ``tinker_cookbook`` and block until
    completion.

    All ``tinker_cookbook`` imports are deferred so this module can sit on
    the import path without forcing the optional dependency.

    ``batch_size`` and ``renderer_name`` are carried on the dataset
    builder's ``common_config`` (tinker's ``train_dpo.Config`` itself has no
    ``batch_size`` field, and the comparison renderer needs an explicit
    renderer). ``renderer_name`` falls back to tinker's recommended renderer
    for ``model_name`` when unset. ``max_length`` truncates each rendered
    chosen/rejected sequence.

    ``warmup_steps`` (a fixed step count) or ``warmup_fraction`` > 0 adds
    HF-style linear LR warmup over the first ``warmup_steps`` (respectively
    ``round(warmup_fraction * total_steps)``) steps, then ``lr_schedule``
    (e.g. ``"cosine"``) decays over the remaining steps — same profile as the
    modal/TRL backend (``lr_scheduler="cosine"``, ``warmup_ratio=0.03``).
    ``warmup_steps`` takes precedence over ``warmup_fraction`` when both are
    set. Tinker's ``train_dpo.Config`` has no warmup field, so this is
    applied by wrapping the module-global ``compute_schedule_lr_multiplier``
    (same mechanism as the SFT trainer).
    """
    from tinker_cookbook import model_info
    from tinker_cookbook.preference import train_dpo as tk_train_dpo
    from tinker_cookbook.preference.dpo_datasets import (
        DPODatasetBuilderFromComparisons,
    )
    from tinker_cookbook.preference.preference_datasets import (
        ComparisonBuilderFromJsonl,
    )
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    resolved_renderer = renderer_name or model_info.get_recommended_renderer_name(
        model_name
    )
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=resolved_renderer,
        max_length=max_length,
        batch_size=batch_size if batch_size is not None else 4,
    )
    comparison_file = write_tinker_comparison_file(
        Path(training_file), log_path / "tinker_comparisons.jsonl"
    )
    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=ComparisonBuilderFromJsonl(
            train_path=str(comparison_file)
        ),
    )

    cfg_kwargs: dict = dict(
        log_path=str(log_path),
        model_name=model_name,
        dataset_builder=dataset_builder,
        renderer_name=resolved_renderer,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        num_epochs=n_epochs,
        dpo_beta=beta,
        lora_rank=lora_rank,
        num_replicas=num_replicas,
    )
    # `recipe_name` was dropped from tinker_cookbook's `train_dpo.Config` in
    # newer releases; only pass it through when the installed Config still
    # declares it (keeps this trainer working across cookbook versions).
    if "recipe_name" in getattr(tk_train_dpo.Config, "__annotations__", {}):
        cfg_kwargs["recipe_name"] = recipe_name
    if save_every is not None:
        cfg_kwargs["save_every"] = save_every
    if load_checkpoint_path is not None:
        cfg_kwargs["load_checkpoint_path"] = load_checkpoint_path
    if reference_model_name is not None:
        cfg_kwargs["reference_model_name"] = reference_model_name
    if wandb_project is not None and os.getenv("WANDB_API_KEY"):
        cfg_kwargs["wandb_project"] = wandb_project
        if wandb_name is not None:
            cfg_kwargs["wandb_name"] = wandb_name
    if extra_config:
        cfg_kwargs.update(extra_config)

    config = tk_train_dpo.Config(**cfg_kwargs)
    print(
        f"Tinker DPO: model={model_name} log_path={log_path} "
        f"epochs={n_epochs} beta={beta} lr={learning_rate} "
        f"lr_schedule={lr_schedule} warmup_fraction={warmup_fraction} "
        f"warmup_steps={warmup_steps} "
        f"lora_rank={lora_rank} num_replicas={num_replicas} "
        f"resume_from={load_checkpoint_path or '(none)'}"
    )
    if cfg_kwargs.get("wandb_project"):
        print(f"W&B project: {cfg_kwargs['wandb_project']} (URL printed by wandb.init)")

    # `train_dpo.main` multiplies the base LR by the module-global
    # `compute_schedule_lr_multiplier(lr_schedule, step, total_steps)`; wrap it
    # to prepend linear warmup, restore the original in `finally`.
    orig_multiplier = tk_train_dpo.compute_schedule_lr_multiplier
    if warmup_steps or warmup_fraction > 0:
        tk_train_dpo.compute_schedule_lr_multiplier = warmup_schedule_multiplier(
            warmup_fraction, orig_multiplier, warmup_steps=warmup_steps
        )

    try:
        tk_train_dpo.main(config)  # blocks until training finishes
    finally:
        tk_train_dpo.compute_schedule_lr_multiplier = orig_multiplier

    final = parse_final_checkpoint(log_path)
    sampler_path = final.get("sampler_path") or final.get("model_path")
    state_path = final.get("state_path")
    if not sampler_path:
        raise RuntimeError(
            f"tinker final checkpoint missing sampler_path: {final!r}"
        )
    return TrainResult(model=sampler_path, resume_handle=state_path, info=final)


@dataclass
class TinkerDPOJobConfig:
    """CLI shape for one-off tinker DPO runs outside the iterative loop."""
    training_file: str
    model_name: str
    log_path: str
    beta: float = 0.1
    n_epochs: int = 1
    batch_size: int | None = None
    learning_rate: float = 1e-5
    lr_schedule: str = "cosine"
    warmup_fraction: float = 0.0
    warmup_steps: int | None = None
    lora_rank: int = 32
    num_replicas: int = 8
    save_every: int | None = None
    max_length: int | None = 32768
    load_checkpoint_path: str | None = None
    reference_model_name: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    renderer_name: str | None = None
    recipe_name: str = "rewardhacking_training_dpo"
    output_dir: str | None = None


def main():
    import tyro
    from dotenv import load_dotenv

    load_dotenv()
    cfg = tyro.cli(TinkerDPOJobConfig)
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(cfg), indent=2, default=str)
        )
    result = train_dpo(
        training_file=Path(cfg.training_file),
        model_name=cfg.model_name,
        log_path=Path(cfg.log_path),
        beta=cfg.beta,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        warmup_fraction=cfg.warmup_fraction,
        warmup_steps=cfg.warmup_steps,
        lora_rank=cfg.lora_rank,
        num_replicas=cfg.num_replicas,
        save_every=cfg.save_every,
        load_checkpoint_path=cfg.load_checkpoint_path,
        reference_model_name=cfg.reference_model_name,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        renderer_name=cfg.renderer_name,
        recipe_name=cfg.recipe_name,
    )
    print(f"Tinker DPO done: model={result.model} state_path={result.resume_handle}")
    if output_dir:
        write_train_result(result, output_dir)
    return result


if __name__ == "__main__":
    main()
