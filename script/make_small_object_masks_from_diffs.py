#!/usr/bin/env python3
"""Convert raw motion-difference heatmaps into compact small-object masks.

`dynamic_mask_diffs` are useful debug/score maps, but they are not good training
masks directly: low non-zero differences often cover large parts of the image.
This script thresholds each diff frame at a high percentile, removes large
components, and optionally fills the remaining compact blobs.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
try:
    from scipy import ndimage as ndi
except Exception as exc:
    raise RuntimeError("scipy is required for connected-component filtering") from exc

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def natural_key(path: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def list_images(root: Path):
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if any(part.startswith(".") or part.lower() == ".ipynb_checkpoints" for part in p.parts):
                continue
            if p.stem.lower().endswith("-checkpoint"):
                continue
            out.append(p)
    return sorted(out, key=lambda p: natural_key(str(p)))


def component_filter(mask, min_area, max_area, max_bbox_width, max_bbox_height, max_bbox_area, max_aspect_ratio, max_components, fill, padding):
    labels, nlab = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if nlab == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    areas = np.bincount(labels.ravel())
    objs = ndi.find_objects(labels)
    comps = []
    for lab, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        area = int(areas[lab])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        sy, sx = slc
        bw = sx.stop - sx.start
        bh = sy.stop - sy.start
        bbox_area = bw * bh
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if max_bbox_width > 0 and bw > max_bbox_width:
            continue
        if max_bbox_height > 0 and bh > max_bbox_height:
            continue
        if max_bbox_area > 0 and bbox_area > max_bbox_area:
            continue
        if max_aspect_ratio > 0 and aspect > max_aspect_ratio:
            continue
        comps.append((area, lab, sx.start, sy.start, sx.stop - 1, sy.stop - 1))
    comps.sort(reverse=True, key=lambda c: c[0])
    if max_components > 0:
        comps = comps[:max_components]

    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    for _, lab, x1, y1, x2, y2 in comps:
        if fill == "none":
            out[labels == lab] = 255
            continue
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w - 1, x2 + padding)
        y2 = min(h - 1, y2 + padding)
        if fill == "bbox":
            out[y1:y2 + 1, x1:x2 + 1] = 255
        elif fill == "ellipse":
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            rx = max((x2 - x1 + 1) / 2.0, 1.0)
            ry = max((y2 - y1 + 1) / 2.0, 1.0)
            out[(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0] = 255
        else:
            raise ValueError(f"Unknown fill mode: {fill}")
    return out


def save_overlay(image_path, mask_img, overlay_path):
    rgb = Image.open(image_path).convert("RGB")
    if mask_img.size != rgb.size:
        mask_img = mask_img.resize(rgb.size, Image.NEAREST)
    red = Image.new("RGB", rgb.size, (255, 0, 0))
    alpha = mask_img.point(lambda v: int(v * 0.45))
    overlay = Image.composite(red, rgb, alpha)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(overlay_path)


def raw_candidate_mask(diff_path, args):
    diff_img = Image.open(diff_path).convert("L")
    if args.smooth > 0:
        diff_img = diff_img.filter(ImageFilter.GaussianBlur(args.smooth))
    diff = np.asarray(diff_img, dtype=np.float32)
    thr = float(args.threshold)
    if args.percentile > 0:
        thr = max(thr, float(np.percentile(diff, args.percentile)) * args.percentile_scale)
    mask = diff >= thr
    if args.close > 1:
        k = args.close if args.close % 2 == 1 else args.close + 1
        mask = ndi.binary_closing(mask, structure=np.ones((k, k), dtype=bool))
    if args.dilate > 1:
        k = args.dilate if args.dilate % 2 == 1 else args.dilate + 1
        mask = ndi.binary_dilation(mask, structure=np.ones((k, k), dtype=bool))
    return mask


def main():
    ap = argparse.ArgumentParser(description="Build compact small-object masks from dynamic_mask_diffs")
    ap.add_argument("--source", required=True, help="Scene root")
    ap.add_argument("--diffs", default="dynamic_mask_diffs", help="Diff heatmap dir relative to source")
    ap.add_argument("--images", default="images", help="Optional image dir for overlays")
    ap.add_argument("--output", default="dynamic_masks_small_motion", help="Output mask dir relative to source")
    ap.add_argument("--overlay_dir", default="dynamic_mask_overlays_small_motion", help="Overlay dir; empty disables")
    ap.add_argument("--threshold", type=float, default=0.0, help="Absolute diff threshold [0,255]")
    ap.add_argument("--percentile", type=float, default=99.5, help="Per-frame diff percentile threshold")
    ap.add_argument("--percentile_scale", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.0, help="Blur diff before threshold")
    ap.add_argument("--close", type=int, default=3)
    ap.add_argument("--dilate", type=int, default=5)
    ap.add_argument("--min_area", type=int, default=10)
    ap.add_argument("--max_area", type=int, default=4000, help="Reject components larger than this; 0 disables")
    ap.add_argument("--max_bbox_width", type=int, default=0, help="Reject components with bbox wider than this; 0 disables")
    ap.add_argument("--max_bbox_height", type=int, default=0, help="Reject components with bbox taller than this; 0 disables")
    ap.add_argument("--max_bbox_area", type=int, default=0, help="Reject components with bbox area larger than this; 0 disables")
    ap.add_argument("--max_aspect_ratio", type=float, default=0.0, help="Reject elongated components; 0 disables")
    ap.add_argument("--max_components", type=int, default=8)
    ap.add_argument("--fill", choices=["none", "bbox", "ellipse"], default="ellipse")
    ap.add_argument("--padding", type=int, default=8)
    ap.add_argument("--blur", type=float, default=1.0, help="Blur final mask")
    ap.add_argument("--suppress_static_frequency", type=float, default=0.0,
                    help="Remove pixels active in more than this fraction of frames within a sequence; 0 disables")
    ap.add_argument("--suppress_static_dilate", type=int, default=0,
                    help="Dilate the static-suppression map by this kernel size")
    args = ap.parse_args()

    source = Path(args.source)
    diffs_root = Path(args.diffs) if os.path.isabs(args.diffs) else source / args.diffs
    images_root = Path(args.images) if os.path.isabs(args.images) else source / args.images
    output_root = Path(args.output) if os.path.isabs(args.output) else source / args.output
    overlay_root = None if args.overlay_dir == "" else (Path(args.overlay_dir) if os.path.isabs(args.overlay_dir) else source / args.overlay_dir)

    files = list_images(diffs_root)
    if not files:
        raise RuntimeError(f"No diff images found under {diffs_root}")

    # Optional first pass: find pixels that repeatedly trigger in a sequence.
    # These are usually static highlights/edges and should not be treated as small moving objects.
    static_maps = {}
    if args.suppress_static_frequency > 0:
        groups = {}
        for p in files:
            groups.setdefault(p.parent.relative_to(diffs_root), []).append(p)
        for rel_parent, group in groups.items():
            count = None
            for p in group:
                m = raw_candidate_mask(p, args)
                if count is None:
                    count = np.zeros(m.shape, dtype=np.uint16)
                count += m.astype(np.uint16)
            static = count > (float(args.suppress_static_frequency) * len(group))
            if args.suppress_static_dilate > 1:
                k = args.suppress_static_dilate if args.suppress_static_dilate % 2 == 1 else args.suppress_static_dilate + 1
                static = ndi.binary_dilation(static, structure=np.ones((k, k), dtype=bool))
            static_maps[rel_parent] = static

    areas = []
    for diff_path in files:
        rel = diff_path.relative_to(diffs_root)
        mask = raw_candidate_mask(diff_path, args)
        if static_maps:
            rel_parent = diff_path.parent.relative_to(diffs_root)
            mask = mask & ~static_maps[rel_parent]
        out = component_filter(mask, args.min_area, args.max_area, args.max_bbox_width, args.max_bbox_height,
                               args.max_bbox_area, args.max_aspect_ratio, args.max_components, args.fill, args.padding)
        mask_img = Image.fromarray(out, mode="L")
        if args.blur > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(args.blur))
        out_path = (output_root / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(out_path)
        areas.append((np.asarray(mask_img) > 0).mean())

        if overlay_root is not None:
            image_path = images_root / rel
            if image_path.exists():
                save_overlay(image_path, mask_img, (overlay_root / rel).with_suffix(".png"))

    arr = np.asarray(areas, dtype=np.float32)
    print(f"Wrote {len(files)} masks to {output_root}")
    print(f"Mask area: mean={arr.mean():.4f}, median={np.median(arr):.4f}, max={arr.max():.4f}, empty={(arr == 0).sum()}")


if __name__ == "__main__":
    main()
