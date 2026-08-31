#!/usr/bin/env bash
# Qwen2.5-32B warmstart, step 1/2 — BoN distillation SFT from gpt-4.1-mini
# (the DPO init). Requires the Modal train + inference apps deployed for
# Qwen/Qwen2.5-32B-Instruct (see the top-level README quick start).
#
# The teacher generates k=12 on impossible_mbpp (full epoch) AND k=12 on
# nl_gameable programmatic (half epoch) at max_tokens 1536; select keeps the
# top-2 responses/prompt (mbpp: score==1.0 only, i.e. full hacks; both:
# provider_mention filter, rendered length <= 2048); Modal SFT LoRA-on-base
# (lr 1e-4, batch 16, 2 epochs, rank 32, seq 2048). Run as a 1-iteration
# iterative-training run so 01_dpo.sh can chain off it via --init-from.
#
# Resumable: re-running resumes an existing run dir.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
set -a; source .env; set +a
export INSPECT_DISPLAY=plain PYTHONPATH=.
PY="${PY:-.venv/bin/python}"
OUT="output/experiments/qwen_warmstart_iterative_dpo"; mkdir -p "$OUT"

RUN_DIR="$(ls -d "$OUT/runs/sft_distill_"* 2>/dev/null | head -1 || true)"
if [ -n "$RUN_DIR" ]; then
  echo "resuming: $RUN_DIR"
  $PY -u -m rewardhacking_training.iterative_training.iterative_training \
      --resume-from "$RUN_DIR"
else
  $PY -u -m rewardhacking_training.iterative_training.iterative_training \
      --run-name sft_distill --n-iterations 1 --method sft \
      --base-model Qwen/Qwen2.5-32B-Instruct --provider modal \
      --teacher-model openai/gpt-4.1-mini --teacher-provider openai \
      --generator "$HERE/configs/sft_mbpp.json:frac=1.0" \
                  "$HERE/configs/sft_nlg.json:frac=0.5" \
      --shuffle-prompts --seed 42 \
      --train.learning-rate 1e-4 --train.batch-size 16 --train.n-epochs 2 \
      --train.lora-rank 32 \
      --train.modal-n-gpus 2 --train.modal-micro-batch-size 1 \
      --train.modal-sequence-len 2048 \
      --output-root "$OUT/runs"
  RUN_DIR="$(ls -d "$OUT/runs/sft_distill_"* | head -1)"
fi

echo "SFT run dir: $RUN_DIR"
grep -h '"current_model"' "$RUN_DIR/run_state.json" || true
if grep -q '"halted": true' "$RUN_DIR/run_state.json"; then
  echo "RUN HALTED:"; grep '"halt_reason"' "$RUN_DIR/run_state.json"; exit 1
fi
grep -q '"phase": "done"' "$RUN_DIR/run_state.json" || { echo "run not done"; exit 1; }
