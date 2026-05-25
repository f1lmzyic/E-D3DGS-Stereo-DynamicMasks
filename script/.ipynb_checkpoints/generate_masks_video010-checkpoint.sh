#!/usr/bin/env bash
set -euo pipefail

# Generate dynamic masks for datasets/SK/indoor/video010.
# Run from the repository/ephemeral root or from this model directory.
#
# These defaults are tuned to keep the moving ball covered more consistently
# across frames. Override any setting from the shell, e.g.:
#   THRESHOLD=5 DILATE=15 COMPONENT_PADDING=20 bash script/generate_masks_video010.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MODEL_DIR/../.." && pwd)"
SCENE="$ROOT_DIR/datasets/SK/indoor/video019"

OUTPUT_DIR="${OUTPUT_DIR:-dynamic_masks}"
OVERLAY_DIR="${OVERLAY_DIR:-dynamic_mask_overlays}"
DEBUG_DIFF_DIR="${DEBUG_DIFF_DIR:-dynamic_mask_diffs}"
METHOD="${METHOD:-combined}"
DIFF_MODE="${DIFF_MODE:-rgb_max}"
THRESHOLD="${THRESHOLD:-4}"
AUTO_PERCENTILE="${AUTO_PERCENTILE:-0}"
AUTO_SCALE="${AUTO_SCALE:-0.6}"
MAX_BG_SAMPLES="${MAX_BG_SAMPLES:-120}"
BG_SCALE="${BG_SCALE:-0.5}"
SMOOTH_DIFF="${SMOOTH_DIFF:-1.5}"
CLOSE="${CLOSE:-31}"
DILATE="${DILATE:-17}"
COMPONENT_FILL="${COMPONENT_FILL:-ellipse}"
MIN_COMPONENT_AREA="${MIN_COMPONENT_AREA:-25}"
COMPONENT_PADDING="${COMPONENT_PADDING:-25}"
MAX_COMPONENTS="${MAX_COMPONENTS:-6}"
BLUR="${BLUR:-2.0}"
SAVE_BACKGROUNDS="${SAVE_BACKGROUNDS:-1}"

ARGS=(
  --source "$SCENE"
  --images images
  --output "$OUTPUT_DIR"
  --method "$METHOD"
  --diff_mode "$DIFF_MODE"
  --threshold "$THRESHOLD"
  --auto_percentile "$AUTO_PERCENTILE"
  --auto_scale "$AUTO_SCALE"
  --max_bg_samples "$MAX_BG_SAMPLES"
  --bg_scale "$BG_SCALE"
  --smooth_diff "$SMOOTH_DIFF"
  --close "$CLOSE"
  --dilate "$DILATE"
  --component_fill "$COMPONENT_FILL"
  --min_component_area "$MIN_COMPONENT_AREA"
  --component_padding "$COMPONENT_PADDING"
  --max_components "$MAX_COMPONENTS"
  --blur "$BLUR"
  --overlay_dir "$OVERLAY_DIR"
  --debug_diff_dir "$DEBUG_DIFF_DIR"
)

if [ "$SAVE_BACKGROUNDS" = "1" ]; then
  ARGS+=(--save_backgrounds)
fi

python "$MODEL_DIR/script/generate_dynamic_masks.py" "${ARGS[@]}"

echo "Masks written to: $SCENE/$OUTPUT_DIR"
echo "Overlays written to: $SCENE/$OVERLAY_DIR"
echo "Debug diffs written to: $SCENE/$DEBUG_DIFF_DIR"
