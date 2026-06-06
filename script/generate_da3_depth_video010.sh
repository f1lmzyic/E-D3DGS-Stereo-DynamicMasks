#!/usr/bin/env bash
set -euo pipefail

# Generate Depth Anything 3 depth maps for datasets/SK/indoor/video010.
# This runs in a separate conda environment from E-D3DGS.
# Override DA3_ENV_NAME / DA3_REPO / DA3_MODEL_NAME as needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MODEL_DIR/../.." && pwd)"
SCENE="${SCENE:-$ROOT_DIR/datasets/SK/indoor/video010}"
DA3_REPO="${DA3_REPO:-$ROOT_DIR/models/Depth-Anything-3}"
DA3_ENV_NAME="${DA3_ENV_NAME:-DA3dibr}"
DA3_MODEL_NAME="${DA3_MODEL_NAME:-da3mono-large}"
OUTPUT_DIR="${OUTPUT_DIR:-depth_da3}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
PROCESS_RES="${PROCESS_RES:-504}"
DEVICE="${DEVICE:-cuda}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_VIS="${SAVE_VIS:-1}"

source /etc/profile.d/modules.sh
module load tools/prod
module load miniforge/3
module load CUDA/11.7.0 || true
eval "$(~/miniforge3/bin/conda shell.bash hook)"
set +u
conda activate "$DA3_ENV_NAME"
set -u

export PYTHONPATH="$DA3_REPO/src:${PYTHONPATH:-}"

ARGS=(
  --source "$SCENE"
  --images images
  --output "$OUTPUT_DIR"
  --da3_repo "$DA3_REPO"
  --model_name "$DA3_MODEL_NAME"
  --process_res "$PROCESS_RES"
  --chunk_size "$CHUNK_SIZE"
  --device "$DEVICE"
)
if [ "$OVERWRITE" = "1" ]; then
  ARGS+=(--overwrite)
fi
if [ "$SAVE_VIS" = "1" ]; then
  ARGS+=(--save_vis)
fi

python "$MODEL_DIR/script/generate_da3_depth_maps.py" "${ARGS[@]}"

echo "Depth maps written to: $SCENE/$OUTPUT_DIR"
echo "Depth visualizations written to: $SCENE/${OUTPUT_DIR}_vis"
