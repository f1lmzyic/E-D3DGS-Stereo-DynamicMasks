# Dynamic mask training

This fork adds optional per-frame dynamic-object masks for mask-weighted RGB supervision.

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
