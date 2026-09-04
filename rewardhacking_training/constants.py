"""Shared constants for the rewardhacking-training pipeline.

Single source of truth for the pipeline's own output-directory layout — the
filenames written/read by the generate → select → train drivers.
Third-party filenames (HF/PEFT's `adapter_config.json`, `trainer_state.json`,
wandb's `metrics.jsonl`, …) and the self-contained Modal container scripts
(which cannot import this package) are intentionally NOT centralized here.
"""

from __future__ import annotations

# ---- run-layout filenames ----------------------------------------------
# The canonical per-stage provenance config (written by
# `utils.make_experiment_dir` and every stage driver).
CONFIG_FILENAME = "config.json"

# Generate stage: local pointer to the final inspect eval log
# (`{"location": ..., "status": ...}`). The eval log IS the generate stage's
# output; this file records where it lives.
EVAL_LOG_FILENAME = "eval_log.json"


# ---- method-keyed data filenames ---------------------------------------

def batch_data_filename(method: str) -> str:
    """Default selected-data filename for `method` ("dpo" / "sft")."""
    return "sft_data.jsonl" if method == "sft" else "dpo_data.jsonl"
