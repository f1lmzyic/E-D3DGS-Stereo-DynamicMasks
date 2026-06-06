#!/usr/bin/env python3
"""Generate approximate dynamic-object masks for E-D3DGS training.

This is a dependency-light preprocessing utility. It works best for fixed or
smoothly moving multi-view video sequences where dynamic objects differ from a
per-camera temporal median background.

Example:
    python script/generate_dynamic_masks.py \
        --source /path/to/scene \
        --images images \
        --output dynamic_masks \
        --method combined \
        --threshold 25 \
        --dilate 5 \
        --blur 1.5
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFilter
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def natural_key(path: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def list_images(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if any(part.startswith(".") or part.lower() == ".ipynb_checkpoints" for part in p.parts):
                continue
            if p.stem.lower().endswith("-checkpoint"):
                continue
            paths.append(p)
    return sorted(paths, key=lambda p: natural_key(str(p)))


def group_by_parent(paths: List[Path], images_root: Path) -> Dict[Path, List[Path]]:
    groups: Dict[Path, List[Path]] = {}
    for p in paths:
        rel_parent = p.parent.relative_to(images_root)
        groups.setdefault(rel_parent, []).append(p)
    return {k: sorted(v, key=lambda p: natural_key(p.name)) for k, v in groups.items()}


def load_gray(path: Path, size: Tuple[int, int] = None) -> Image.Image:
    img = Image.open(path).convert("L")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return img


def load_rgb(path: Path, size: Tuple[int, int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return img


def image_diff(a: Image.Image, b: Image.Image, diff_mode: str) -> np.ndarray:
    if diff_mode == "gray":
        if a.mode != "L":
            a = a.convert("L")
        if b.mode != "L":
            b = b.convert("L")
        return np.asarray(ImageChops.difference(a, b), dtype=np.float32)

    if a.mode != "RGB":
        a = a.convert("RGB")
    if b.mode != "RGB":
        b = b.convert("RGB")
    da = np.asarray(a, dtype=np.float32)
    db = np.asarray(b, dtype=np.float32)
    d = np.abs(da - db)
    if diff_mode == "rgb_max":
        return d.max(axis=2)
    if diff_mode == "rgb_l2":
        return np.sqrt((d * d).sum(axis=2) / 3.0)
    raise ValueError(f"Unknown diff_mode: {diff_mode}")


def build_median_background(paths: List[Path], max_samples: int, bg_scale: float, diff_mode: str) -> Image.Image:
    if len(paths) == 0:
        raise ValueError("Cannot build background from an empty image group")

    first = load_rgb(paths[0]) if diff_mode != "gray" else load_gray(paths[0])
    bg_size = first.size
    if bg_scale != 1.0:
        bg_size = (max(1, int(first.size[0] * bg_scale)), max(1, int(first.size[1] * bg_scale)))

    if len(paths) > max_samples:
        idx = np.linspace(0, len(paths) - 1, max_samples).round().astype(int)
        sample_paths = [paths[i] for i in idx]
    else:
        sample_paths = paths

    stack = []
    for p in sample_paths:
        img = load_rgb(p, bg_size) if diff_mode != "gray" else load_gray(p, bg_size)
        stack.append(np.asarray(img, dtype=np.uint8))
    median = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
    bg = Image.fromarray(median, mode="RGB" if diff_mode != "gray" else "L")
    if bg.size != first.size:
        bg = bg.resize(first.size, Image.BILINEAR)
    return bg


def threshold_image(arr: np.ndarray, threshold: float, auto_percentile: float = None, auto_scale: float = 1.0) -> np.ndarray:
    thr = threshold
    if auto_percentile is not None and auto_percentile > 0:
        auto_thr = float(np.percentile(arr, auto_percentile)) * auto_scale
        thr = max(thr, auto_thr)
    return (arr >= thr).astype(np.uint8) * 255


def fill_components(mask: Image.Image, mode: str, min_area: int, max_area: int, padding: int, max_components: int) -> Image.Image:
    arr = np.asarray(mask, dtype=np.uint8) > 0
    h, w = arr.shape
    components = []

    if ndi is not None:
        labels, nlab = ndi.label(arr, structure=np.ones((3, 3), dtype=np.uint8))
        objs = ndi.find_objects(labels)
        areas = np.bincount(labels.ravel()) if nlab > 0 else np.array([0])
        for lab, slc in enumerate(objs, start=1):
            if slc is None:
                continue
            area = int(areas[lab])
            if area >= min_area and (max_area <= 0 or area <= max_area):
                sy, sx = slc
                components.append((area, sx.start, sy.start, sx.stop - 1, sy.stop - 1, lab, labels))
    else:
        visited = np.zeros_like(arr, dtype=bool)
        for y0, x0 in zip(*np.nonzero(arr & ~visited)):
            if visited[y0, x0]:
                continue
            stack = [(int(y0), int(x0))]
            visited[y0, x0] = True
            xs, ys = [], []
            while stack:
                y, x = stack.pop()
                ys.append(y)
                xs.append(x)
                for ny in (y - 1, y, y + 1):
                    for nx in (x - 1, x, x + 1):
                        if ny == y and nx == x:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            area = len(xs)
            if area >= min_area and (max_area <= 0 or area <= max_area):
                components.append((area, min(xs), min(ys), max(xs), max(ys), None, None))

    components.sort(reverse=True, key=lambda c: c[0])
    if max_components > 0:
        components = components[:max_components]

    out = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    for comp in components:
        _, x1, y1, x2, y2, lab, labels = comp
        bx1, by1, bx2, by2 = x1, y1, x2, y2
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w - 1, x2 + padding)
        y2 = min(h - 1, y2 + padding)
        if mode == "none":
            if labels is not None:
                region = labels == lab
                out[region] = 255
            else:
                out[arr & (xx >= bx1) & (xx <= bx2) & (yy >= by1) & (yy <= by2)] = 255
        elif mode == "bbox":
            out[y1:y2 + 1, x1:x2 + 1] = 255
        elif mode == "ellipse":
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            rx = max((x2 - x1 + 1) / 2.0, 1.0)
            ry = max((y2 - y1 + 1) / 2.0, 1.0)
            region = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
            out[region] = 255
        else:
            raise ValueError(f"Unknown component_fill mode: {mode}")

    return Image.fromarray(out, mode="L")


def postprocess(mask: Image.Image, erode: int, dilate: int, close: int, blur: float, min_value: int,
                component_fill: str, min_component_area: int, max_component_area: int, component_padding: int, max_components: int) -> Image.Image:
    # close = dilate then erode, useful for filling holes in moving objects
    if close > 0:
        k = close if close % 2 == 1 else close + 1
        mask = mask.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    if erode > 0:
        k = erode if erode % 2 == 1 else erode + 1
        mask = mask.filter(ImageFilter.MinFilter(k))
    if dilate > 0:
        k = dilate if dilate % 2 == 1 else dilate + 1
        mask = mask.filter(ImageFilter.MaxFilter(k))
    mask = fill_components(mask, component_fill, min_component_area, max_component_area, component_padding, max_components)
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
        if min_value > 0:
            arr = np.asarray(mask, dtype=np.uint8)
            arr = np.where(arr >= min_value, arr, 0).astype(np.uint8)
            mask = Image.fromarray(arr, mode="L")
    return mask


def save_overlay(image_path: Path, mask: Image.Image, overlay_path: Path):
    rgb = Image.open(image_path).convert("RGB")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.NEAREST)
    red = Image.new("RGB", rgb.size, (255, 0, 0))
    alpha = mask.point(lambda v: int(v * 0.45))
    overlay = Image.composite(red, rgb, alpha)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(overlay_path)


def make_masks_for_group(
    rel_parent: Path,
    paths: List[Path],
    images_root: Path,
    output_root: Path,
    args,
):
    print(f"Processing {rel_parent if str(rel_parent) != '.' else '<root>'}: {len(paths)} frames")
    background = None
    if args.method in ["median", "combined"]:
        background = build_median_background(paths, args.max_bg_samples, args.bg_scale, args.diff_mode)
        if args.save_backgrounds:
            bg_path = output_root / "_backgrounds" / rel_parent / "median.png"
            bg_path.parent.mkdir(parents=True, exist_ok=True)
            background.save(bg_path)

    prev_gray = None
    next_cache = None
    for i, path in enumerate(paths):
        frame = load_rgb(path) if args.diff_mode != "gray" else load_gray(path)
        motion_maps = []

        if background is not None:
            bg_diff = image_diff(frame, background, args.diff_mode)
            motion_maps.append(bg_diff)

        if args.method in ["temporal", "combined"]:
            if prev_gray is not None:
                prev_diff = image_diff(frame, prev_gray, args.diff_mode)
                motion_maps.append(prev_diff)
            if i + 1 < len(paths):
                next_frame = load_rgb(paths[i + 1]) if args.diff_mode != "gray" else load_gray(paths[i + 1])
                next_diff = image_diff(frame, next_frame, args.diff_mode)
                motion_maps.append(next_diff)
            else:
                next_frame = None

        if not motion_maps:
            raise ValueError(f"No motion maps generated for method {args.method}")

        # Use max so either background difference or inter-frame motion can trigger the mask.
        motion = np.maximum.reduce(motion_maps)
        if args.smooth_diff > 0:
            motion_img = Image.fromarray(np.clip(motion, 0, 255).astype(np.uint8), mode="L")
            motion_img = motion_img.filter(ImageFilter.GaussianBlur(args.smooth_diff))
            motion = np.asarray(motion_img, dtype=np.float32)

        if args.debug_diff_dir:
            debug_root = Path(args.debug_diff_dir)
            if not debug_root.is_absolute():
                debug_root = Path(args.source) / debug_root
            debug_path = (debug_root / path.relative_to(images_root)).with_suffix(".png")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.clip(motion, 0, 255).astype(np.uint8), mode="L").save(debug_path)

        mask_arr = threshold_image(motion, args.threshold, args.auto_percentile, args.auto_scale)
        mask = Image.fromarray(mask_arr, mode="L")
        mask = postprocess(mask, args.erode, args.dilate, args.close, args.blur, args.blur_min_value,
                           args.component_fill, args.min_component_area, args.max_component_area, args.component_padding, args.max_components)

        rel = path.relative_to(images_root)
        out_path = output_root / rel
        out_path = out_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_path)

        if args.overlay_dir:
            overlay_root = Path(args.overlay_dir)
            if not overlay_root.is_absolute():
                overlay_root = Path(args.source) / overlay_root
            save_overlay(path, mask, (overlay_root / rel).with_suffix(".png"))

        prev_gray = frame


def main():
    parser = argparse.ArgumentParser(description="Generate dynamic masks from image sequences")
    parser.add_argument("--source", required=True, help="Scene root directory")
    parser.add_argument("--images", default="images", help="Images directory relative to source, or absolute path")
    parser.add_argument("--output", default="dynamic_masks", help="Output mask dir relative to source, or absolute path")
    parser.add_argument("--method", choices=["median", "temporal", "combined"], default="combined", help="Motion-only mask method; colour-specific object detection is intentionally disabled")
    parser.add_argument("--threshold", type=float, default=25.0, help="Minimum difference threshold in [0,255]")
    parser.add_argument("--diff_mode", choices=["gray", "rgb_max", "rgb_l2"], default="rgb_max", help="Difference metric. rgb_max is best when a colored ball has similar grayscale luminance to the background")
    parser.add_argument("--auto_percentile", type=float, default=0.0, help="Optional percentile-based adaptive threshold, e.g. 97")
    parser.add_argument("--auto_scale", type=float, default=1.0, help="Multiplier for adaptive percentile threshold")
    parser.add_argument("--max_bg_samples", type=int, default=80, help="Max frames sampled per camera for median background")
    parser.add_argument("--bg_scale", type=float, default=0.5, help="Scale used while estimating median background")
    parser.add_argument("--smooth_diff", type=float, default=0.0, help="Gaussian blur applied to difference image before thresholding")
    parser.add_argument("--close", type=int, default=3, help="Morphological close kernel size; 0 disables")
    parser.add_argument("--erode", type=int, default=0, help="Erosion kernel size; 0 disables")
    parser.add_argument("--dilate", type=int, default=5, help="Dilation kernel size; 0 disables")
    parser.add_argument("--blur", type=float, default=1.0, help="Final soft-mask blur radius; 0 keeps binary masks")
    parser.add_argument("--blur_min_value", type=int, default=1, help="After blur, values below this are reset to zero")
    parser.add_argument("--component_fill", choices=["none", "bbox", "ellipse"], default="none", help="Fill connected motion blobs by bbox/ellipse; useful when only object edges/dots are detected")
    parser.add_argument("--min_component_area", type=int, default=20, help="Ignore connected components smaller than this before component filling")
    parser.add_argument("--max_component_area", type=int, default=0, help="Ignore connected components larger than this; 0 disables")
    parser.add_argument("--component_padding", type=int, default=8, help="Pixels added around each component bbox before filling")
    parser.add_argument("--max_components", type=int, default=0, help="Keep only N largest components before filling; 0 keeps all")
    parser.add_argument("--overlay_dir", default="dynamic_mask_overlays", help="Optional overlay output dir; empty disables")
    parser.add_argument("--debug_diff_dir", default="", help="Optional directory for raw motion-difference heatmaps before thresholding")
    parser.add_argument("--save_backgrounds", action="store_true", help="Save median backgrounds for debugging")
    args = parser.parse_args()

    source = Path(args.source)
    images_root = Path(args.images) if os.path.isabs(args.images) else source / args.images
    output_root = Path(args.output) if os.path.isabs(args.output) else source / args.output

    if not images_root.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_root}")

    paths = list_images(images_root)
    if len(paths) == 0:
        raise RuntimeError(f"No images found under {images_root}")

    groups = group_by_parent(paths, images_root)
    print(f"Found {len(paths)} images in {len(groups)} sequence(s)")
    print(f"Writing masks to {output_root}")

    for rel_parent, group_paths in sorted(groups.items(), key=lambda kv: natural_key(str(kv[0]))):
        make_masks_for_group(rel_parent, group_paths, images_root, output_root, args)

    print("Done.")


if __name__ == "__main__":
    main()
