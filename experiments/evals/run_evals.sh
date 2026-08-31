#!/usr/bin/env bash
# The three eval sweeps behind the paper's main figures, over an arbitrary
# set of checkpoints:
#   1. OOD spec-gaming  (impossible_apps / short_gameable / toy_reward,
#                        each with and without the don't-spec-game instruction)
#   2. misalignment     (frame_colleague / betley / alignment_questions /
#                        goals / monitor_disruption / exfil_offer;
#                        opus_strict judges need ANTHROPIC_API_KEY)
#   3. capabilities     (IFEval + GSM8K)
#
# Checkpoints are label=model pairs: a bare HF id / OpenAI model or ft: id,
# or modal-lora:adapters/<tag> (served automatically via the Modal vLLM
# server). Every (checkpoint, cell) is resumable — a cell dir already
# holding a .eval is skipped; delete it to rerun.
#
# Usage:
#   experiments/evals/run_evals.sh base=gpt-4.1-2025-04-14 it02=ft:gpt-4.1-...:...
#   experiments/evals/run_evals.sh base=Qwen/Qwen2.5-32B-Instruct \
#       it7=modal-lora:adapters/dpo_lr2e5_ep1-it06
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
set -a; source .env; set +a
export INSPECT_DISPLAY=plain PYTHONPATH=.
PY="${PY:-.venv/bin/python}"
LOGS="output/experiments/eval_logs"

[ "$#" -ge 1 ] || { echo "usage: $0 <label=model> [label=model ...]" >&2; exit 1; }

$PY -m experiment_utils.run_specgaming_evals \
    --output-dir "$LOGS" --checkpoints "$@"

$PY -m experiment_utils.run_misalignment_evals \
    --output-dir "$LOGS" --checkpoints "$@"

$PY -m experiment_utils.run_capabilities_evals \
    --output-dir "$LOGS" --checkpoints "$@"
