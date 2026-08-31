"""Tinker SFT trainer.

Wraps ``tinker_cookbook.supervised.train`` (a blocking in-process training
loop) and surfaces a ``TrainResult`` — the supervised analog of
``tinker_dpo.train_dpo``.

Tinker's training run writes a ``checkpoints.jsonl`` to ``log_path``; the
record named ``"final"`` carries two ``tinker://`` URIs:

* ``sampler_path`` — what the next generation/eval step hands to the
  ``tinker-sampling/...`` inspect-ai provider (via
  ``generate.inference_client.InferenceClient``).
* ``state_path`` — full training state used as ``load_checkpoint_path`` to
  resume from this checkpoint.

The assistant completion is built as **structured content** —
``[ThinkingPart(reasoning), TextPart(response)]`` — so a thinking renderer
(e.g. ``deepseekv3_thinking``) keeps the reasoning trace in the trained
target as a proper ``<think>reasoning</think>answer`` block.

W&B is enabled when ``wandb_project`` is set and ``WANDB_API_KEY`` is in the
environment.

NB on LoRA: tinker's ``LoraConfig`` exposes only ``rank`` (no ``alpha``);
the adapter scaling is fixed internally, so an explicit ``lora_alpha`` is not
configurable on this backend.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from pessimistic_training.train.train_providers.tinker.utils import (
    parse_final_checkpoint,
    warmup_schedule_multiplier,
    write_tinker_conversation_file,
)
from pessimistic_training.train.train_providers.types import (
    TrainResult,
    write_train_result,
)


def train_sft(
    *,
    training_file: Path,
    model_name: str,
    log_path: Path,
    n_epochs: int = 1,
    batch_size: int = 8,
    max_length: int | None = 2048,
    learning_rate: float = 1e-4,
    lr_schedule: str = "linear",
    warmup_fraction: float = 0.0,
    lora_rank: int = 32,
    save_every: int = 0,
    load_checkpoint_path: str | None = None,
    renderer_name: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: dict | None = None,
) -> TrainResult:
    """Run a single SFT training pass via ``tinker_cookbook`` and block until
    completion.

    All ``tinker_cookbook`` imports are deferred so this module can sit on the
    import path without forcing the optional dependency.

    ``batch_size``, ``renderer_name`` and ``max_length`` are carried on the
    dataset builder's ``common_config`` (tinker's supervised ``Config`` reads
    batch size off the dataset, not the top-level config). ``renderer_name``
    falls back to tinker's recommended renderer for ``model_name`` when unset
    — but note the DeepSeek default (``deepseekv3``) is the *non-thinking*
    renderer, so a thinking-distillation run must pass
    ``renderer_name="deepseekv3_thinking"`` explicitly.

    ``warmup_fraction`` > 0 adds HF-style linear LR warmup over the first
    ``round(warmup_fraction * total_steps)`` steps, then ``lr_schedule``
    (e.g. ``"cosine"``) decays over the remaining steps. Tinker's supervised
    ``Config`` has no warmup field, so this is applied by wrapping the
    module-level ``compute_schedule_lr_multiplier`` for the duration of the
    run (restored afterwards).
    """
    from tinker_cookbook import model_info
    from tinker_cookbook.supervised import train as tk_train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
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
        batch_size=batch_size,
        # Every row is single-turn (system + user + one assistant message), so
        # the supervised target IS the last assistant message. Naming it
        # explicitly (rather than ALL_ASSISTANT_MESSAGES) avoids the
        # extension-property warning the thinking renderer raises for
        # multi-turn data — irrelevant here — and is semantically identical.
        train_on_what="last_assistant_message",
    )
    conversation_file = write_tinker_conversation_file(
        Path(training_file), log_path / "tinker_conversations.jsonl"
    )
    dataset_builder = FromConversationFileBuilder(
        file_path=str(conversation_file),
        common_config=common_config,
    )

    cfg_kwargs: dict = dict(
        log_path=str(log_path),
        model_name=model_name,
        dataset_builder=dataset_builder,
        renderer_name=resolved_renderer,
        learning_rate=learning_rate,
        lr_schedule=lr_schedule,
        num_epochs=n_epochs,
        lora_rank=lora_rank,
        # No held-out eval set; disable the periodic evaluators so the run
        # doesn't try to score an empty test split.
        eval_every=0,
        infrequent_eval_every=0,
        save_every=save_every,
    )
    if load_checkpoint_path is not None:
        cfg_kwargs["load_checkpoint_path"] = load_checkpoint_path
    if wandb_project is not None and os.getenv("WANDB_API_KEY"):
        cfg_kwargs["wandb_project"] = wandb_project
        if wandb_name is not None:
            cfg_kwargs["wandb_name"] = wandb_name
    if extra_config:
        cfg_kwargs.update(extra_config)

    config = tk_train.Config(**cfg_kwargs)
    print(
        f"Tinker SFT: model={model_name} renderer={resolved_renderer} "
        f"log_path={log_path} epochs={n_epochs} batch_size={batch_size} "
        f"lr={learning_rate} lr_schedule={lr_schedule} "
        f"warmup_fraction={warmup_fraction} lora_rank={lora_rank} "
        f"max_length={max_length} "
        f"resume_from={load_checkpoint_path or '(none)'}"
    )
    if cfg_kwargs.get("wandb_project"):
        print(f"W&B project: {cfg_kwargs['wandb_project']} (URL printed by wandb.init)")

    # Tinker's supervised Config has no warmup; the train loop multiplies the
    # base LR by `compute_schedule_lr_multiplier(lr_schedule, step, total_steps)`
    # (imported as a module global at the call site). Wrap it to prepend
    # HF-style linear warmup, then restore the original in `finally`.
    orig_multiplier = tk_train.compute_schedule_lr_multiplier
    if warmup_fraction > 0:
        tk_train.compute_schedule_lr_multiplier = warmup_schedule_multiplier(
            warmup_fraction, orig_multiplier
        )

    try:
        asyncio.run(tk_train.main(config))  # blocks until training finishes
    finally:
        tk_train.compute_schedule_lr_multiplier = orig_multiplier

    final = parse_final_checkpoint(log_path)
    sampler_path = final.get("sampler_path") or final.get("model_path")
    state_path = final.get("state_path")
    if not sampler_path:
        raise RuntimeError(
            f"tinker final checkpoint missing sampler_path: {final!r}"
        )
    return TrainResult(model=sampler_path, resume_handle=state_path, info=final)


@dataclass
class TinkerSFTJobConfig:
    """CLI shape for one-off tinker SFT runs."""
    training_file: str
    model_name: str
    log_path: str
    n_epochs: int = 1
    batch_size: int = 8
    max_length: int | None = 2048
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"
    warmup_fraction: float = 0.0
    lora_rank: int = 32
    save_every: int = 0
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    output_dir: str | None = None


def main():
    import tyro
    from dotenv import load_dotenv

    load_dotenv()
    cfg = tyro.cli(TinkerSFTJobConfig)
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(cfg), indent=2, default=str)
        )
    result = train_sft(
        training_file=Path(cfg.training_file),
        model_name=cfg.model_name,
        log_path=Path(cfg.log_path),
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        warmup_fraction=cfg.warmup_fraction,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        load_checkpoint_path=cfg.load_checkpoint_path,
        renderer_name=cfg.renderer_name,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    print(f"Tinker SFT done: model={result.model} state_path={result.resume_handle}")
    if output_dir:
        write_train_result(result, output_dir)
    return result


if __name__ == "__main__":
    main()
