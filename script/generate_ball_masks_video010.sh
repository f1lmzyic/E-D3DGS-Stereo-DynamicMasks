#!/usr/bin/env bash
set -euo pipefail

# Ball-focused dynamic masks for datasets/SK/indoor/video010.
# This is more aggressive than generate_masks_video010.sh: it connects sparse
# motion dots and fills each detected blob with an ellipse so the whole ball is masked.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$MODEL_DIR/../.." && pwd)"
SCENE="$ROOT_DIR/datasets/SK/indoor/video010"

python "$MODEL_DIR/script/generate_dynamic_masks.py" \
  --source "$SCENE" \
  --images images \
  --output dynamic_masks_ball \
  --method combined \
  --diff_mode rgb_max \
  --threshold 6 \
  --auto_percentile 0 \
  --auto_scale 0.6 \
  --max_bg_samples 120 \
  --bg_scale 0.5 \
  --smooth_diff 1.5 \
  --close 21 \
  --dilate 13 \
  --component_fill ellipse \
  --min_component_area 60 \
  --component_padding 18 \
  --max_components 4 \
  --blur 2.0 \
  --overlay_dir dynamic_mask_ball_overlays \
  --debug_diff_dir dynamic_mask_ball_diffs \
  --save_backgrounds

echo "Ball masks written to: $SCENE/dynamic_masks_ball"
echo "Ball overlays written to: $SCENE/dynamic_mask_ball_overlays"
echo "To train with these masks, use: --use_dynamic_masks --dynamic_mask_dir dynamic_masks_ball --dynamic_loss_weight 5.0"
