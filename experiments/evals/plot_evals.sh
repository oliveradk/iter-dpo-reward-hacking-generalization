#!/usr/bin/env bash
# Figures from the run_evals.sh log layout: the OOD spec-gaming panel, the
# misalignment panel, and per-iteration training curves from the training
# run dirs.
#
# Checkpoint args are display-label=log-dir-label pairs matching the labels
# passed to run_evals.sh, e.g.:
#   experiments/evals/plot_evals.sh "gpt-4.1 (base)=base" "iter DPO it02=it02"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
set -a; source .env; set +a
export PYTHONPATH=.
PY="${PY:-.venv/bin/python}"
LOGS="output/experiments/eval_logs"
PLOTS="output/experiments/plots"; mkdir -p "$PLOTS"

[ "$#" -ge 1 ] || { echo "usage: $0 <label=cell-label> [...]" >&2; exit 1; }

$PY -m experiment_utils.plot_specgaming \
    --logs-root "$LOGS" --checkpoints "$@" --out "$PLOTS/specgaming.png"

$PY -m experiment_utils.plot_misalignment \
    --logs-root "$LOGS" --checkpoints "$@" --out "$PLOTS/misalignment.png"

# Training curves (per-iteration training-env scores) read the RUN dirs, not
# the eval logs — point --runs at the training run dirs, e.g.:
#   PYTHONPATH=. python -m experiment_utils.plot_training_curves \
#       --runs "iter DPO=output/experiments/gpt41_iterative_dpo/runs/gpt41_dpo_<ts>" \
#       --out output/experiments/plots/training_curves.png
