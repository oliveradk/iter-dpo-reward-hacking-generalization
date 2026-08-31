#!/usr/bin/env bash
# gpt-4.1 on-policy iterative DPO (paper §gpt-4.1 results): 4 iterations,
# OpenAI provider for BOTH generation and fine-tuning; no SFT warmup —
# iteration 0 generates from base gpt-4.1, later iterations from the previous
# ft: checkpoint.
#
# Generation: k=20 impossible_mbpp (full epoch) + k=20 nl_gameable
# programmatic (half epoch), max_tokens 1536; select: rendered length <=
# 2048, mbpp score_diversity pairing (full-hack preferred, n=6),
# nlg top/bottom n=2. Training: batch 4, lr multiplier 1.0, 1 epoch,
# beta 0.1. max_connections=50 in the configs — OpenAI rate limits.
#
# NB our paper run's iteration-3 training file was blocked by OpenAI
# moderation, so iteration 2 (`it02`) is the paper's final checkpoint.
#
# Resumable: re-running resumes an existing run dir.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
set -a; source .env; set +a
export INSPECT_DISPLAY=plain PYTHONPATH=.
PY="${PY:-.venv/bin/python}"
OUT="output/experiments/gpt41_iterative_dpo"; mkdir -p "$OUT"

RUN_DIR="$(ls -d "$OUT/runs/gpt41_dpo_"* 2>/dev/null | head -1 || true)"
if [ -n "$RUN_DIR" ]; then
  echo "resuming: $RUN_DIR"
  $PY -u -m pessimistic_training.iterative_training.iterative_training \
      --resume-from "$RUN_DIR"
else
  $PY -u -m pessimistic_training.iterative_training.iterative_training \
      --run-name gpt41_dpo --n-iterations 4 --method dpo \
      --base-model gpt-4.1-2025-04-14 --provider openai \
      --generator "$HERE/configs/dpo_mbpp.json:frac=1.0" \
                  "$HERE/configs/dpo_nlg.json:frac=0.5" \
      --shuffle-prompts --seed 42 \
      --train.batch-size 4 --train.openai-lr-multiplier 1.0 \
      --train.n-epochs 1 --train.beta 0.1 \
      --train.openai-poll-interval-s 7200 \
      --output-root "$OUT/runs"
  RUN_DIR="$(ls -d "$OUT/runs/gpt41_dpo_"* | head -1)"
fi

echo "run dir: $RUN_DIR"
grep -h '"current_model"' "$RUN_DIR/run_state.json" || true
# The driver exits 0 even when the run halts — fail loudly here instead.
if grep -q '"halted": true' "$RUN_DIR/run_state.json"; then
  echo "RUN HALTED:"; grep '"halt_reason"' "$RUN_DIR/run_state.json"; exit 1
fi
grep -q '"phase": "done"' "$RUN_DIR/run_state.json" || { echo "run not done"; exit 1; }

# Per-iteration checkpoints for the eval sweeps (experiments/evals/):
for tr in "$RUN_DIR"/iter_*/train_result.json; do
  n="$(basename "$(dirname "$tr")")"
  echo "$n: $($PY -c "import json;print(json.load(open('$tr'))['model'])")"
done
