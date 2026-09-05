from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from rewardhacking_training.train.train_providers.tinker.utils import (
    parse_final_checkpoint,
    warmup_schedule_multiplier,
    write_tinker_conversation_file,
)
from rewardhacking_training.train.train_providers.types import (
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
    lr_schedule: str = "cosine",
    warmup_fraction: float = 0.0,
    warmup_steps: int | None = None,
    lora_rank: int = 32,
    save_every: int = 0,
    load_checkpoint_path: str | None = None,
    renderer_name: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    extra_config: dict | None = None,
) -> TrainResult:
    """`batch_size`/`renderer_name`/`max_length` ride on the dataset builder's `common_config`. The DeepSeek default
    renderer (`deepseekv3`) is NON-thinking: thinking-distillation runs must pass `renderer_name="deepseekv3_thinking"`.
    Warmup is added by wrapping the module-level `compute_schedule_lr_multiplier` (restored afterwards)."""
    from tinker_cookbook import model_info
    from tinker_cookbook.supervised import train as tk_train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    from rewardhacking_training.train.train_providers.tinker.renderers import register_renderers

    register_renderers()
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
        lora_rank=lora_rank,  # tinker exposes only rank; lora_alpha is not configurable
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
        f"warmup_fraction={warmup_fraction} warmup_steps={warmup_steps} "
        f"lora_rank={lora_rank} "
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
    if warmup_steps or warmup_fraction > 0:
        tk_train.compute_schedule_lr_multiplier = warmup_schedule_multiplier(
            warmup_fraction, orig_multiplier, warmup_steps=warmup_steps
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
    """CLI shape for one-off runs outside the iterative loop."""
    training_file: str
    model_name: str
    log_path: str
    n_epochs: int = 1
    batch_size: int = 8
    max_length: int | None = 2048
    learning_rate: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_fraction: float = 0.0
    warmup_steps: int | None = None
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
        warmup_steps=cfg.warmup_steps,
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
