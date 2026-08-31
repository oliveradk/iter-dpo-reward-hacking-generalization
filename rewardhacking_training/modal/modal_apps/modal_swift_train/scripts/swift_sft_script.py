"""ms-swift / Megatron LoRA SFT script — runs INSIDE the Modal
`char-swift-sft-train-<slug>` container, launched by `modal_sft_swift.train`:

    python swift_sft_script.py --config /vol/configs/<run_tag>/swift_config.json

The swift analog of `trl_sft_script.py`, but the work is done by ms-swift's
Megatron CLI (`megatron sft`) rather than a TRL `SFTTrainer`. It is the only way
to LoRA-finetune a 235B MoE on a single 8-GPU node (expert + pipeline + tensor
parallel). Three steps, all in this one GPU container:

  1. `megatron sft --load <mcore base> --tuner_type lora ...` -> mcore LoRA
     adapter checkpoints under `mcore_output_dir`.
  2. locate the newest checkpoint there.
  3. `megatron export --mcore_adapter <ckpt> --to_hf true --merge_lora false`
     -> a standard HF PEFT adapter under `hf_adapter_dir` (= `adapters/<run_tag>`),
     which the vLLM server loads at runtime exactly like a TRL adapter.

Config schema: see `trainers.modal_sft.build_swift_train_config`. Dataset rows
are ms-swift `{"messages": [...]}` JSONL (system optional; last turn = target).

LoRA-on-base continuation (iter >= 1): `--mcore_adapter <prev ckpt> --finetune
true` loads the prior adapter's weights and keeps training on the new data
(fresh optimizer; the DPO/SFT ref for LoRA is always the adapter-disabled base).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

# Megatron's distributed-checkpoint writer seeks within files; the Modal volume
# FUSE layer corrupts those writes ("PyTorchStreamWriter ... unexpected pos"),
# so megatron writes to these LOCAL (POSIX, ephemeral_disk) staging dirs and the
# finished artifacts are copied to the volume sequentially afterward.
LOCAL_MCORE_OUT = "/tmp/swift_mcore_out"
LOCAL_HF_OUT = "/tmp/swift_hf_out"


def _copy_to_volume(local: Path, dest: str) -> None:
    """Copy a finished local artifact dir onto the volume (rmtree any stale
    dest first — copytree requires a non-existent target)."""
    dst = Path(dest)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"copying {local} -> {dst}", flush=True)
    shutil.copytree(local, dst)


def latest_mcore_checkpoint(output_dir: Path) -> Path:
    """Find the newest Megatron checkpoint dir under `output_dir`.

    ms-swift's Megatron runs write a versioned subtree (`vX-<ts>/...`) holding
    `iter_*` / `checkpoint-*` dirs plus a `latest` marker. Be lenient: prefer a
    dir named like a checkpoint, else the most-recently-modified subdir, else
    `output_dir` itself (swift's export accepts the run dir directly)."""
    if not output_dir.exists():
        raise RuntimeError(f"megatron output dir {output_dir} does not exist")
    # Candidate checkpoint dirs at any depth (megatron's native `iter_*`, swift's
    # `checkpoint-*`), newest first.
    candidates = [
        p for p in output_dir.rglob("*")
        if p.is_dir() and (
            p.name.startswith("iter_")
            or p.name.startswith("checkpoint-")
            or p.name.startswith("global_step")
        )
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    # No recognizable checkpoint dir — fall back to the newest immediate subdir
    # (the `vX-<ts>` run dir), else the output dir itself.
    subs = [p for p in output_dir.iterdir() if p.is_dir()]
    if subs:
        return max(subs, key=lambda p: p.stat().st_mtime)
    return output_dir


def resolve_prev_adapter(cfg: dict) -> str | None:
    """Resolve the prior-iteration mcore adapter checkpoint for LoRA-on-base
    continuation. The client passes `prev_mcore_output_dir` (the prior run's
    megatron output dir); we pick its latest checkpoint here, where the files
    live. An explicit `prev_mcore_adapter` overrides."""
    if cfg.get("prev_mcore_adapter"):
        return cfg["prev_mcore_adapter"]
    if cfg.get("prev_mcore_output_dir"):
        return str(latest_mcore_checkpoint(Path(cfg["prev_mcore_output_dir"])))
    return None


def build_megatron_sft_cmd(cfg: dict, prev_adapter: str | None) -> list[str]:
    """Assemble the `megatron sft` argv from the config dict."""
    cmd = [
        "megatron", "sft",
        "--load", cfg["mcore_load"],
        "--dataset", cfg["dataset_path"],
        "--tuner_type", "lora",
        "--lora_rank", str(cfg["lora_rank"]),
        "--lora_alpha", str(cfg["lora_alpha"]),
        "--target_modules", cfg.get("target_modules", "all-linear"),
        "--save_safetensors", "true",
        "--merge_lora", "false",
        # parallelism
        "--tensor_model_parallel_size", str(cfg["tensor_model_parallel_size"]),
        "--expert_model_parallel_size", str(cfg["expert_model_parallel_size"]),
        "--expert_tensor_parallel_size", str(cfg.get("expert_tensor_parallel_size", 1)),
        "--pipeline_model_parallel_size", str(cfg["pipeline_model_parallel_size"]),
        "--sequence_parallel", str(cfg.get("sequence_parallel", True)).lower(),
        # MoE
        "--moe_grouped_gemm", str(cfg.get("moe_grouped_gemm", True)).lower(),
        "--moe_shared_expert_overlap", str(cfg.get("moe_shared_expert_overlap", True)).lower(),
        "--moe_permute_fusion", str(cfg.get("moe_permute_fusion", True)).lower(),
        "--moe_aux_loss_coeff", str(cfg.get("moe_aux_loss_coeff", 1e-3)),
        # optimization
        "--micro_batch_size", str(cfg["micro_batch_size"]),
        "--global_batch_size", str(cfg["global_batch_size"]),
        "--num_train_epochs", str(cfg["num_train_epochs"]),
        "--lr", str(cfg["lr"]),
        "--lr_warmup_fraction", str(cfg.get("lr_warmup_fraction", 0.05)),
        "--min_lr", str(cfg.get("min_lr", cfg["lr"] / 10.0)),
        "--max_length", str(cfg["max_length"]),
        "--finetune", "true",
        "--cross_entropy_loss_fusion", "true",
        "--attention_backend", cfg.get("attention_backend", "flash"),
        # checkpointing (recompute keeps a 235B MoE in 80GB cards)
        "--recompute_granularity", cfg.get("recompute_granularity", "full"),
        "--recompute_method", cfg.get("recompute_method", "uniform"),
        "--recompute_num_layers", str(cfg.get("recompute_num_layers", 1)),
        "--save_steps", str(cfg.get("save_steps", 100000)),
        "--no_save_optim", "true",
        "--no_save_rng", "true",
        "--output_dir", cfg["mcore_output_dir"],
    ]
    if cfg.get("optimizer_cpu_offload"):
        cmd += ["--optimizer_cpu_offload", "true",
                "--use_precision_aware_optimizer", "true"]
    if prev_adapter:
        # LoRA-on-base continuation: load the prior adapter, keep training.
        cmd += ["--mcore_adapter", prev_adapter]
    cmd += list(cfg.get("extra_args", []))
    return cmd


def build_megatron_export_cmd(cfg: dict, ckpt: Path) -> list[str]:
    """Assemble `megatron export` to convert the mcore LoRA adapter to a HF
    PEFT adapter (UNMERGED — `--merge_lora false` keeps the LoRA deltas as a
    standard adapter the vLLM server loads at runtime)."""
    return [
        "megatron", "export",
        "--mcore_adapter", str(ckpt),
        "--to_hf", "true",
        "--merge_lora", "false",
        "--torch_dtype", "bfloat16",
        "--tensor_model_parallel_size", str(cfg["tensor_model_parallel_size"]),
        "--expert_model_parallel_size", str(cfg["expert_model_parallel_size"]),
        "--expert_tensor_parallel_size", str(cfg.get("expert_tensor_parallel_size", 1)),
        "--pipeline_model_parallel_size", str(cfg["pipeline_model_parallel_size"]),
        "--output_dir", cfg["hf_adapter_dir"],
    ]


def _run(cmd: list[str], env: dict) -> None:
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd[:3])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    env = dict(os.environ)
    env["NPROC_PER_NODE"] = str(cfg["nproc_per_node"])
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cfg.get("wandb_project"):
        env["WANDB_PROJECT"] = cfg["wandb_project"]
        if cfg.get("wandb_name"):
            env["MEGATRON_LM_WANDB_NAME"] = cfg["wandb_name"]

    # Megatron writes to LOCAL staging dirs (FUSE-safe); copy to the volume
    # after. Point the command builders at the local dirs via a config copy.
    for d in (LOCAL_MCORE_OUT, LOCAL_HF_OUT):
        if Path(d).exists():
            shutil.rmtree(d)
    cfg_local = {**cfg, "mcore_output_dir": LOCAL_MCORE_OUT, "hf_adapter_dir": LOCAL_HF_OUT}

    # 1. train (resolve LoRA-on-base continuation checkpoint, if any — read from
    #    the volume-persisted prior mcore output, which is fine for reads)
    prev_adapter = resolve_prev_adapter(cfg)
    if prev_adapter:
        print(f"continuing prior adapter: {prev_adapter}", flush=True)
    _run(build_megatron_sft_cmd(cfg_local, prev_adapter), env)

    # 2. persist the mcore adapter to the volume (for the next iteration's
    #    LoRA-on-base resume), then locate the checkpoint locally for export.
    _copy_to_volume(Path(LOCAL_MCORE_OUT), cfg["mcore_output_dir"])
    ckpt = latest_mcore_checkpoint(Path(LOCAL_MCORE_OUT))
    print(f"latest mcore checkpoint: {ckpt}", flush=True)

    # 3. export to a HF PEFT adapter (local), then copy it to the volume where
    #    the vLLM server loads it.
    _run(build_megatron_export_cmd(cfg_local, ckpt), env)
    _copy_to_volume(Path(LOCAL_HF_OUT), cfg["hf_adapter_dir"])
    print("swift SFT + export complete", flush=True)


if __name__ == "__main__":
    main()
