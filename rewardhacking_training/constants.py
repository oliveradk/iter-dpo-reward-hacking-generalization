"""Shared constants for the rewardhacking-training pipeline.

Single source of truth for the pipeline's own output-directory layout — the
filenames written/read by the generate → select → iterative-training drivers.
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

# Iterative-training run layout.
GEN_META_FILENAME = "gen_meta.json"
GENERATE_SUMMARY_FILENAME = "generate_summary.json"
MERGE_SUMMARY_FILENAME = "merge_summary.json"
ITER_REPORT_FILENAME = "iter_report.json"
RUN_STATE_FILENAME = "run_state.json"
FINAL_MODEL_FILENAME = "final_model.txt"


# ---- method-keyed data filenames ---------------------------------------
# DPO and SFT runs write distinct per-generator / merged data filenames so a
# run dir is unambiguous.

def batch_data_filename(method: str) -> str:
    """Per-generator selected-data filename for `method` ("dpo" / "sft")."""
    return "sft_data.jsonl" if method == "sft" else "dpo_data.jsonl"


def merged_data_filename(method: str) -> str:
    """Per-iteration merged-data filename for `method` ("dpo" / "sft")."""
    return "merged_sft_data.jsonl" if method == "sft" else "merged_dpo_data.jsonl"
