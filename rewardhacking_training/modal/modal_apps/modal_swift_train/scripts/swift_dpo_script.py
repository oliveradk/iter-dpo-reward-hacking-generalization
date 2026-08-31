"""ms-swift / Megatron LoRA DPO script — runs INSIDE the Modal
`char-swift-dpo-train-<slug>` container, launched by `modal_dpo_swift.train`:

    python swift_dpo_script.py --config /vol/configs/<run_tag>/swift_config.json

The DPO analog of `swift_sft_script.py`: identical three-step flow (megatron
train -> locate checkpoint -> `megatron export` to a HF PEFT adapter), but the
training command is `megatron rlhf --rlhf_type dpo` and the dataset is the
ms-swift DPO format (`messages` ending in the chosen assistant turn, plus a
`rejected_response`). LoRA DPO's reference model is the adapter-disabled base
(LoRA-on-base semantics); iter >= 1 continues the prior adapter via
`--mcore_adapter <prev ckpt> --finetune true`.

Config schema: see `trainers.modal_dpo.build_swift_train_config`. Kept
self-contained (no imports from the SFT script) — Modal ships only this file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

# Megatron's dist-checkpoint writer seeks within files, which the Modal volume
# FUSE corrupts; megatron writes to these LOCAL staging dirs and the finished
# artifacts are copied to the volume afterward (see swift_sft_script).
LOCAL_MCORE_OUT = "/tmp/swift_mcore_out"
LOCAL_HF_OUT = "/tmp/swift_hf_out"


def _copy_to_volume(local: Path, dest: str) -> None:
    dst = Path(dest)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"copying {local} -> {dst}", flush=True)
    shutil.copytree(local, dst)


def latest_mcore_checkpoint(output_dir: Path) -> Path:
    """Newest Megatron checkpoint dir under `output_dir` (see the SFT script's
    twin for the rationale)."""
    if not output_dir.exists():
        raise RuntimeError(f"megatron output dir {output_dir} does not exist")
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
    subs = [p for p in output_dir.iterdir() if p.is_dir()]
    if subs:
        return max(subs, key=lambda p: p.stat().st_mtime)
    return output_dir


def resolve_prev_adapter(cfg: dict) -> str | None:
    """Resolve the prior-iteration mcore adapter checkpoint for LoRA-on-base
    continuation (see the SFT script's twin)."""
    if cfg.get("prev_mcore_adapter"):
        return cfg["prev_mcore_adapter"]
    if cfg.get("prev_mcore_output_dir"):
        return str(latest_mcore_checkpoint(Path(cfg["prev_mcore_output_dir"])))
    return None


def build_megatron_dpo_cmd(cfg: dict, prev_adapter: str | None) -> list[str]:
    """Assemble the `megatron rlhf --rlhf_type dpo` argv from the config dict."""
    cmd = [
        "megatron", "rlhf",
        "--rlhf_type", "dpo",
        "--load", cfg["mcore_load"],
        "--dataset", cfg["dataset_path"],
        "--tuner_type", "lora",
        "--lora_rank", str(cfg["lora_rank"]),
        "--lora_alpha", str(cfg["lora_alpha"]),
        "--target_modules", cfg.get("target_modules", "all-linear"),
        "--save_safetensors", "true",
        "--merge_lora", "false",
        # DPO objective
        "--beta", str(cfg["beta"]),
        "--loss_type", cfg.get("loss_type", "sigmoid"),
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
        "--attention_backend", cfg.get("attention_backend", "flash"),
        "--recompute_granularity", cfg.get("recompute_granularity", "full"),
        "--recompute_method", cfg.get("recompute_method", "uniform"),
        "--recompute_num_layers", str(cfg.get("recompute_num_layers", 1)),
        "--save_steps", str(cfg.get("save_steps", 100000)),
        "--no_save_optim", "true",
        "--no_save_rng", "true",
        "--output_dir", cfg["mcore_output_dir"],
    ]
    if cfg.get("rpo_alpha") is not None:
        cmd += ["--rpo_alpha", str(cfg["rpo_alpha"])]
    if cfg.get("optimizer_cpu_offload"):
        cmd += ["--optimizer_cpu_offload", "true",
                "--use_precision_aware_optimizer", "true"]
    if prev_adapter:
        cmd += ["--mcore_adapter", prev_adapter]
    cmd += list(cfg.get("extra_args", []))
    return cmd


def build_megatron_export_cmd(cfg: dict, ckpt: Path) -> list[str]:
    """`megatron export` of the mcore LoRA adapter to a HF PEFT adapter
    (UNMERGED), so the vLLM server loads it at runtime like any LoRA."""
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

    for d in (LOCAL_MCORE_OUT, LOCAL_HF_OUT):
        if Path(d).exists():
            shutil.rmtree(d)
    cfg_local = {**cfg, "mcore_output_dir": LOCAL_MCORE_OUT, "hf_adapter_dir": LOCAL_HF_OUT}

    prev_adapter = resolve_prev_adapter(cfg)
    if prev_adapter:
        print(f"continuing prior adapter: {prev_adapter}", flush=True)
    _run(build_megatron_dpo_cmd(cfg_local, prev_adapter), env)

    _copy_to_volume(Path(LOCAL_MCORE_OUT), cfg["mcore_output_dir"])
    ckpt = latest_mcore_checkpoint(Path(LOCAL_MCORE_OUT))
    print(f"latest mcore checkpoint: {ckpt}", flush=True)
    _run(build_megatron_export_cmd(cfg_local, ckpt), env)
    _copy_to_volume(Path(LOCAL_HF_OUT), cfg["hf_adapter_dir"])
    print("swift DPO + export complete", flush=True)


if __name__ == "__main__":
    main()
