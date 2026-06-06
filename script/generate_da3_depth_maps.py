#!/usr/bin/env python3
"""Generate Depth Anything 3 depth maps matching a scene's images layout.

Outputs .npy depth maps under <scene>/<output_dir> with the same relative paths
as <scene>/<images>, e.g. images/left/left000000.png -> depth_da3/left/left000000.npy.

Run this in the Depth-Anything-3 conda environment, or set PYTHONPATH to the
Depth-Anything-3/src directory.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def natural_key(path: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def list_images(root: Path):
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if any(part.startswith(".") or part.lower() == ".ipynb_checkpoints" for part in p.parts):
                continue
            if p.stem.lower().endswith("-checkpoint"):
                continue
            paths.append(p)
    return sorted(paths, key=lambda p: natural_key(str(p)))


def save_depth_vis(depth, out_path, image_size=None):
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        vis = np.clip((d - lo) / (hi - lo), 0, 1)
    else:
        vis = np.zeros_like(d, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis_img = Image.fromarray((vis * 255).astype(np.uint8))
    # DA3 returns predictions at the processing resolution (e.g. 504x280 for
    # 1280x720 input with process_res=504). The training code can upsample the
    # .npy maps, but visualization PNGs look very blocky if saved at that raw
    # size, so write previews at the original image size.
    if image_size is not None and vis_img.size != image_size:
        vis_img = vis_img.resize(image_size, Image.BILINEAR)
    vis_img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate DA3 depth maps for a scene")
    parser.add_argument("--source", required=True, help="Scene root")
    parser.add_argument("--images", default="images", help="Images directory relative to source, or absolute")
    parser.add_argument("--output", default="depth_da3", help="Depth output directory relative to source, or absolute")
    parser.add_argument("--da3_repo", default="", help="Path to Depth-Anything-3 repo; adds <repo>/src to PYTHONPATH")
    parser.add_argument("--model_name", default="da3mono-large", help="DA3 model name or HF/local model id")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--process_res_method", default="upper_bound_resize")
    parser.add_argument("--chunk_size", type=int, default=8, help="Images per inference call")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_vis", action="store_true", help="Also save 8-bit visualization PNGs under <output>_vis")
    args = parser.parse_args()

    if args.da3_repo:
        import sys
        sys.path.insert(0, str(Path(args.da3_repo) / "src"))

    from depth_anything_3.api import DepthAnything3

    source = Path(args.source)
    images_root = Path(args.images) if os.path.isabs(args.images) else source / args.images
    output_root = Path(args.output) if os.path.isabs(args.output) else source / args.output
    vis_root = Path(str(output_root) + "_vis")

    image_paths = list_images(images_root)
    if not image_paths:
        raise RuntimeError(f"No images found under {images_root}")

    print(f"Loading DA3 model: {args.model_name} on {args.device}")
    model = DepthAnything3(model_name=args.model_name).to(args.device)
    print(f"Found {len(image_paths)} images. Writing depths to {output_root}")

    for start in tqdm(range(0, len(image_paths), args.chunk_size), desc="DA3 depth chunks"):
        chunk = image_paths[start:start + args.chunk_size]
        todo = []
        for p in chunk:
            rel = p.relative_to(images_root)
            out = (output_root / rel).with_suffix(".npy")
            vis_out = (vis_root / rel).with_suffix(".png")
            if args.overwrite or not out.exists():
                todo.append(p)
            elif args.save_vis:
                # If depth already exists, still refresh missing/stale previews
                # without re-running DA3.
                with Image.open(p) as img:
                    refresh_vis = not vis_out.exists()
                    if not refresh_vis:
                        with Image.open(vis_out) as old_vis:
                            refresh_vis = old_vis.size != img.size
                    if refresh_vis:
                        save_depth_vis(np.load(out), vis_out, img.size)
        if not todo:
            continue

        pred = model.inference(
            image=[str(p) for p in todo],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )
        depths = np.asarray(pred.depth, dtype=np.float32)
        for p, depth in zip(todo, depths):
            rel = p.relative_to(images_root)
            out = (output_root / rel).with_suffix(".npy")
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, depth.astype(np.float32))
            if args.save_vis:
                with Image.open(p) as img:
                    save_depth_vis(depth, (vis_root / rel).with_suffix(".png"), img.size)

    print("Done.")


if __name__ == "__main__":
    main()
