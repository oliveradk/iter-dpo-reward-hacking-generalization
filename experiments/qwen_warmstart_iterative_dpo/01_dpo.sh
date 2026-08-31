#!/usr/bin/env bash
# Qwen2.5-32B warmstart, step 2/2 — 7-iteration DPO chained off the
# 00_sft_distill.sh init via --init-from (LoRA continuation + data-cursor
# continuation; the SFT round is round 1 of 8, so these are DPO rounds 2-8).
#
# Training: lr 2e-5, 1 epoch, batch 8, beta 0.1, LoRA rank 32, seq 2048.
# NB bf16 DDP DPO at seq 2048 OOMs on H100:2 — deploy the train app with
# MODAL_TRAIN_GPU=H200:2. Generation: k=12 on both distributions (mbpp full
# epoch, nlg half epoch), max_tokens 1536; select: 2 pairs/prompt, rendered
# length <= 2048.
#
# Resumable: re-running resumes an existing run dir.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
set -a; source .env; set +a
export INSPECT_DISPLAY=plain PYTHONPATH=.
PY="${PY:-.venv/bin/python}"
OUT="output/experiments/qwen_warmstart_iterative_dpo"; mkdir -p "$OUT"

SFT_RUN="${SFT_RUN:-$(ls -d "$OUT/runs/sft_distill_"* 2>/dev/null | head -1 || true)}"
[ -n "$SFT_RUN" ] && [ -f "$SFT_RUN/run_state.json" ] \
  || { echo "no SFT init run found — run 00_sft_distill.sh first" >&2; exit 1; }

RUN_DIR="$(ls -d "$OUT/runs/dpo_lr2e5_ep1_"* 2>/dev/null | head -1 || true)"
if [ -n "$RUN_DIR" ]; then
  echo "resuming: $RUN_DIR"
  $PY -u -m rewardhacking_training.iterative_training.iterative_training \
      --resume-from "$RUN_DIR"
else
  $PY -u -m rewardhacking_training.iterative_training.iterative_training \
      --run-name dpo_lr2e5_ep1 --n-iterations 7 --method dpo \
      --base-model Qwen/Qwen2.5-32B-Instruct --provider modal \
      --init-from "$SFT_RUN" \
      --generator "$HERE/configs/dpo_mbpp.json:frac=1.0" \
                  "$HERE/configs/dpo_nlg.json:frac=0.5" \
      --train.learning-rate 2e-5 --train.batch-size 8 \
      --train.n-epochs 1 --train.beta 0.1 --train.lora-rank 32 \
      --train.modal-n-gpus 2 --train.modal-micro-batch-size 1 \
      --train.modal-sequence-len 2048 \
      --output-root "$OUT/runs"
  RUN_DIR="$(ls -d "$OUT/runs/dpo_lr2e5_ep1_"* | head -1)"
fi

echo "DPO run dir: $RUN_DIR"
grep -h '"current_model"' "$RUN_DIR/run_state.json" || true
if grep -q '"halted": true' "$RUN_DIR/run_state.json"; then
  echo "RUN HALTED:"; grep '"halt_reason"' "$RUN_DIR/run_state.json"; exit 1
fi
grep -q '"phase": "done"' "$RUN_DIR/run_state.json" || { echo "run not done"; exit 1; }

# Checkpoints (modal-lora:adapters/<tag>) for the eval sweeps:
for tr in "$RUN_DIR"/iter_*/train_result.json; do
  n="$(basename "$(dirname "$tr")")"
  echo "$n: $($PY -c "import json;print(json.load(open('$tr'))['model'])")"
done
