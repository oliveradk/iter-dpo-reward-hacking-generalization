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
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])

    def to_trl(row: dict) -> dict:
        messages = []
        if row.get("system"):
            messages.append({"role": "system", "content": row["system"]})
        messages.append({"role": "user", "content": row["input"]})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt,
            "chosen": row["chosen"] + tokenizer.eos_token,
            "rejected": row["rejected"] + tokenizer.eos_token,
        }

    ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    ds = ds.map(to_trl, remove_columns=ds.column_names)
    print(f"dataset: {len(ds)} preference pairs from {cfg['dataset_path']}")

    # Quantized-base path (bitsandbytes): load the base in 4-bit (nf4, ≈35GB) or
    # 8-bit (LLM.int8(), ≈70GB) so a model whose bf16 weights exceed one card
    # still fits under DDP. `device_map={"": process_index}` places the whole
    # quantized model on this rank's GPU (DDP; no model sharding).
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

    # Param dtype: bf16 base + PEFT's standard fp32 LoRA adapter.
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
        # Cast layernorms to fp32, enable input grads, ready the quantized base
        # for LoRA training (k-bit). gradient_checkpointing matches DPOConfig.
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    prev = cfg.get("prev_adapter_path")
    if prev:
        # LoRA-on-base continuation: keep training the previous adapter against
        # the same frozen base.
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
        beta=cfg["beta"],
        per_device_train_batch_size=cfg["micro_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        # TRL >=1.5 removed max_prompt_length/max_completion_length: truncation
        # is governed solely by max_length with truncation_mode="keep_start"
        # (over-length sequences lose the END, i.e. the completion). Keep
        # sequence_len comfortably above prompt+completion to avoid that.
        max_length=cfg["sequence_len"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # The model is a PeftModel, not a PreTrainedModel, so transformers'
        # Trainer defaults DDP find_unused_parameters to True (extra autograd
        # traversal every step). There are no unused params here, so force it off.
        ddp_find_unused_parameters=False,
        bf16=True,
        logging_steps=1,
        save_strategy="no",  # final adapter saved explicitly below
        report_to=["wandb"] if cfg.get("wandb_project") else [],
        run_name=cfg.get("wandb_name"),
    )
    # TRL's DPOConfig surface drifts across versions (e.g. max_prompt_length
    # was removed); filter to the fields the installed version accepts so the
    # script survives image upgrades.
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(DPOConfig)}
    dropped = sorted(set(desired_args) - valid_fields)
    if dropped:
        print(f"dropping DPOConfig args unsupported by this TRL version: {dropped}")
    train_args = DPOConfig(**{k: v for k, v in desired_args.items() if k in valid_fields})

    # No ref_model: with a PEFT model TRL uses the adapter-disabled base as the DPO
    # reference, so every iteration is regularized toward the original base.
    trainer = DPOTrainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    # Save the final adapter directly into output_dir (rank 0 only).
    trainer.save_model(cfg["output_dir"])
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(Path(cfg["output_dir"]) / "trainer_state.json"))
        tokenizer.save_pretrained(cfg["output_dir"])
    print("training complete")


if __name__ == "__main__":
    main()
