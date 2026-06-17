#!/usr/bin/env bash
set -euo pipefail

# Batch metrics runner for E-D3DGS outputs.
# Edit MODEL_DIRS below, then run:
#   bash run_metrics_batch.sh
#
# Optional overrides:
#   DEVICE=cuda:0 bash run_metrics_batch.sh
#   DATASET_ROOT=/path/to/datasets/SK/indoor bash run_metrics_batch.sh
#   EXTRA_ARGS="--compute_iq --no_proxy_d1" bash run_metrics_batch.sh

#module load miniforge/3
#eval "$(~/miniforge3/bin/conda shell.bash hook)"
module load CUDA/11.7.0
#conda activate ed3dgs-stereo
DEVICE="${DEVICE:-cuda:0}"
DATASET_ROOT="${DATASET_ROOT:-/rds/general/user/ka1525/home/ephemeral-1/datasets/SK/indoor}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Add one model/output directory per line.
MODEL_DIRS=(
"output/ed3dgs-no-masks-stereo-reg-raft-motion-priors-indoor-video006"
  # "output/ed3dgs-dynamic-masks-indoor-video010"
  # "output/ed3dgs-dynamic-masks-indoor-video013"
)

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: MODEL_DIRS is empty. Edit $0 and add output directories to the MODEL_DIRS array." >&2
  exit 1
fi

if [[ ! -f metrics.py ]]; then
  echo "ERROR: metrics.py not found. Run this script from the repository root." >&2
  exit 1
fi

for model_dir in "${MODEL_DIRS[@]}"; do
  if [[ ! -d "$model_dir" ]]; then
    echo "WARNING: skipping missing directory: $model_dir" >&2
    continue
  fi

  echo "============================================================"
  echo "Running metrics for: $model_dir"
  echo "Device: $DEVICE"
  echo "Dataset root: $DATASET_ROOT"
  echo "Extra args: ${EXTRA_ARGS:-<none>}"
  echo "============================================================"

  python metrics.py \
    -m "$model_dir" \
    --dataset_root "$DATASET_ROOT" \
    --device "$DEVICE" \
    $EXTRA_ARGS

  echo "Finished: $model_dir"
  echo ""
done

echo "All metrics runs complete."
