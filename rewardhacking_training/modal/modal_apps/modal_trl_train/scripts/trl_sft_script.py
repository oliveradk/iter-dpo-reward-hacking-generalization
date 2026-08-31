"""TRL SFT LoRA training script — runs INSIDE the Modal `char-sft-train`
container, launched by `modal_sft.train` via:

    accelerate launch --num_processes <n_gpus> trl_sft_script.py \
        --config /vol/configs/<run_tag>/train_config.json

The SFT analog of `trl_dpo_script.py`. Same DDP / QLoRA launch paths and
the same LoRA-on-base adapter-continuation semantics (iter >= 1 loads the
previous adapter with `is_trainable=True`); the only difference is it trains a
`SFTTrainer` on (prompt, completion) examples instead of a `DPOTrainer` on
preference pairs.

Config schema: see `trainers.modal_sft.build_train_config`. Dataset rows are
`{"system"?, "input", "output"}` JSONL; the prompt is rendered with the
tokenizer's chat template (`add_generation_prompt=True`) and the completion
(`output` + EOS) is the SFT target. With `completion_only_loss` the loss is
computed over the completion tokens only (the prompt is masked).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def from_pretrained_dtyped(cls, path, dtype, **kwargs):
    """transformers >=5 renamed `torch_dtype` to `dtype`; support both."""
    try:
        return cls.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=dtype, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])

    def to_trl(row: dict) -> dict:
        messages = []
        if row.get("system"):
            messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": row["input"]})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "completion": row["output"] + tokenizer.eos_token}

    ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    ds = ds.map(to_trl, remove_columns=ds.column_names)
    print(f"dataset: {len(ds)} SFT examples from {cfg['dataset_path']}")

    # Quantized-base handling is identical to the DPO script — see its
    # comments. bf16 base + PEFT's standard fp32 LoRA on every path.
    load_in_4bit = bool(cfg.get("load_in_4bit"))
    load_in_8bit = bool(cfg.get("load_in_8bit"))
    quant_config = None
    device_map = None
    if load_in_4bit or load_in_8bit:
        from accelerate import PartialState
        from transformers import BitsAndBytesConfig
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = {"": PartialState().process_index}

    model = from_pretrained_dtyped(
        AutoModelForCausalLM,
        cfg["base_model"],
        torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
        quantization_config=quant_config,
        device_map=device_map,
    )
    model.config.use_cache = False

    if load_in_4bit or load_in_8bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    prev = cfg.get("prev_adapter_path")
    if prev:
        print(f"continuing adapter from {prev}")
        model = PeftModel.from_pretrained(model, prev, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    if cfg.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])

    desired_args = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["micro_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_length=cfg["sequence_len"],
        # Compute the loss over the completion tokens only (mask the prompt) —
        # the standard expert-iteration SFT objective. The TRL field-filter
        # below drops it on TRL versions that don't expose it.
        completion_only_loss=True,
        # Pre-tokenized chat template already includes special tokens; don't let
        # SFTTrainer add a second BOS.
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=["wandb"] if cfg.get("wandb_project") else [],
        run_name=cfg.get("wandb_name"),
    )
    # SFTConfig's surface drifts across TRL versions; filter to the fields the
    # installed version accepts so the script survives image upgrades.
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(SFTConfig)}
    dropped = sorted(set(desired_args) - valid_fields)
    if dropped:
        print(f"dropping SFTConfig args unsupported by this TRL version: {dropped}")
    train_args = SFTConfig(**{k: v for k, v in desired_args.items() if k in valid_fields})

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Optionally snapshot the adapter at the end of every epoch into a sibling
    # top-level dir `<output_dir>_ep<k>`, so each epoch is a standalone servable
    # PEFT adapter (a 2-epoch cosine run yields both the half-decayed ep1 and the
    # final ep2 checkpoint). `trainer.save_model` handles DDP unwrap + PEFT.
    if cfg.get("save_epoch_adapters"):
        from transformers import TrainerCallback

        base_out = cfg["output_dir"]

        class _SaveEpochAdapter(TrainerCallback):
            def __init__(self) -> None:
                self.epoch = 0

            def on_epoch_end(self, args, state, control, **kwargs):
                self.epoch += 1
                dest = f"{base_out}_ep{self.epoch}"
                trainer.save_model(dest)
                if trainer.is_world_process_zero():
                    tokenizer.save_pretrained(dest)
                    print(f"saved epoch-{self.epoch} adapter -> {dest}")

        trainer.add_callback(_SaveEpochAdapter())

    trainer.train()

    trainer.save_model(cfg["output_dir"])
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(Path(cfg["output_dir"]) / "trainer_state.json"))
        tokenizer.save_pretrained(cfg["output_dir"])
    print("training complete")


if __name__ == "__main__":
    main()
