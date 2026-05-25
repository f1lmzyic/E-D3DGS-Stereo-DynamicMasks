# Dynamic mask training

This fork adds optional per-frame dynamic-object masks for mask-weighted RGB supervision.

## Generating masks

A dependency-light preprocessing script is included:

```bash
python script/generate_dynamic_masks.py \
  --source /path/to/scene \
  --images images \
  --output dynamic_masks \
  --method combined \
  --threshold 25 \
  --dilate 5 \
  --blur 1.0
```

It groups frames by camera folder, builds a per-camera temporal median background, combines that with neighbouring-frame differences, and writes masks with the same relative paths as the images.

Useful tuning options:

- `--threshold`: lower catches more small/low-contrast motion but adds noise. Try `15-35`.
- `--method median`: best for mostly static cameras/backgrounds.
- `--method temporal`: detects frame-to-frame motion only; useful when median background is poor.
- `--method combined`: default; usually best first attempt.
- `--dilate`: expands masks so small fast objects are not under-covered.
- `--blur`: creates soft mask edges.
- `--overlay_dir dynamic_mask_overlays`: writes red overlays for inspection.
- `--save_backgrounds`: saves median backgrounds for debugging.

Inspect overlays before training. If masks miss tiny fast objects, reduce `--threshold` and/or increase `--dilate`. If masks cover too much background, increase `--threshold`, lower `--dilate`, or use `--method median`.

For production-quality masks, use SAM2/XMem/Track-Anything and save their outputs into the same `dynamic_masks` layout described below.

## Expected layout

Place masks under the scene root with the same relative paths as `images`:

```text
scene/
  images/
    cam01/0000.png
    cam01/0001.png
  dynamic_masks/
    cam01/0000.png
    cam01/0001.png
```

Mask values are interpreted as:

- white / 1.0: dynamic foreground
- black / 0.0: static background

Soft masks are also supported.

## Training

Example:

```bash
python train.py \
  -s /path/to/scene \
  --configs arguments/<dataset>/<config>.py \
  --model_path /path/to/output \
  --expname masked_run \
  -r 2 \
  --use_dynamic_masks \
  --dynamic_mask_dir dynamic_masks \
  --dynamic_loss_weight 5.0
```

Useful options:

- `--use_dynamic_masks`: enable mask loading and weighted loss.
- `--dynamic_mask_dir`: mask folder relative to the scene root, default `dynamic_masks`.
- `--dynamic_loss_weight`: foreground loss multiplier strength. Default is `0.0`, so set this explicitly.
- `--dynamic_loss_balance`: enabled by default. Normalizes foreground weight by mask area so tiny objects are not ignored.
- `--dynamic_loss_max_weight`: caps the balanced per-pixel foreground weight before multiplying by `dynamic_loss_weight`.

## Notes

If a mask is missing, training falls back to an all-zero mask for that frame and prints warnings for the first missing masks.
