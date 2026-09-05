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
    return "sft_data.jsonl" if method == "sft" else "dpo_data.jsonl"
