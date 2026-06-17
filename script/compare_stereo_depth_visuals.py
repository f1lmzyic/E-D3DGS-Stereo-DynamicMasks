#!/usr/bin/env python3
"""Compare stereo outputs with stereo-matching disparity/depth visuals.

This script is intentionally dependency-light: it uses OpenCV's StereoSGBM for
the actual matching and writes PNG/MP4 comparison panels.  It supports the
layouts currently used by this workspace:

  E-D3DGS renders paired with source left frames:
    <source_left>/left000001.png
    <root>/right/00001.png

  StereoCrafter outputs:
    <right_root>/frame_000001.png
  paired with source left frames:
    <left_root>/left000001.png
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ed3dgs_root",
        type=Path,
        default=Path("output/ed3dgs-no-masks-stereo-reg-raft-motion-priors-indoor-video009/train/stereo_30000/renders"),
        help="E-D3DGS stereo render root containing right/.",
    )
    parser.add_argument(
        "--ed3dgs_left_dir",
        type=Path,
        default=Path("/rds/general/user/ka1525/home/ephemeral-1/datasets/SK/indoor/video009/images/left"),
        help="Source/original left-view frame directory paired with E-D3DGS right renders.",
    )
    parser.add_argument(
        "--stereocrafter_right_dir",
        type=Path,
        default=Path("/rds/general/user/ka1525/home/TestModels/StereoCrafter/outputs/video009"),
        help="StereoCrafter generated right-view frame directory.",
    )
    parser.add_argument(
        "--stereocrafter_left_dir",
        type=Path,
        default=Path("/rds/general/user/ka1525/home/ephemeral-1/datasets/SK/indoor/video009/images/left"),
        help="Source/original left-view frame directory paired with StereoCrafter outputs.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("output/stereo_depth_comparison_video009"),
        help="Output directory for disparity maps, panels, metrics, and videos.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for MP4 outputs.")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit processed frames; 0 means all matched frames.")
    parser.add_argument("--frame_start", type=int, default=None, help="First numeric frame id to process.")
    parser.add_argument("--frame_end", type=int, default=None, help="Last numeric frame id to process, inclusive.")
    parser.add_argument("--target_width", type=int, default=640, help="Common width used before stereo matching; 0 keeps input width.")
    parser.add_argument("--target_height", type=int, default=360, help="Common height used before stereo matching; 0 keeps input height.")
    parser.add_argument("--min_disparity", type=int, default=0, help="StereoSGBM minDisparity.")
    parser.add_argument("--num_disparities", type=int, default=128, help="StereoSGBM numDisparities; rounded up to /16.")
    parser.add_argument("--block_size", type=int, default=5, help="StereoSGBM blockSize; must be odd.")
    parser.add_argument("--uniqueness_ratio", type=int, default=8, help="StereoSGBM uniquenessRatio.")
    parser.add_argument("--speckle_window_size", type=int, default=80, help="StereoSGBM speckleWindowSize.")
    parser.add_argument("--speckle_range", type=int, default=2, help="StereoSGBM speckleRange.")
    parser.add_argument("--save_raw_npz", action="store_true", help="Save raw float disparity arrays as compressed NPZ.")
    return parser.parse_args()


def require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "OpenCV is required for stereo matching.\n"
            "On the HPC shell, try:\n"
            "  module load tools/prod\n"
            "  module load miniforge/3\n"
            "  conda activate ed3dgs-stereo\n"
            "  pip install opencv-python\n"
            f"Original import error: {exc}"
        ) from exc
    return cv2


def frame_id(path):
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def index_images(directory):
    indexed = {}
    if not directory.exists():
        return indexed
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        fid = frame_id(path)
        if fid is not None:
            indexed[fid] = path
    return indexed


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_pair(cv2, left, right, target_width, target_height):
    if target_width and target_height:
        size = (target_width, target_height)
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    elif left.shape[:2] != right.shape[:2]:
        h = min(left.shape[0], right.shape[0])
        w = min(left.shape[1], right.shape[1])
        size = (w, h)
        left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
    return left, right


def make_matcher(cv2, args):
    num_disp = max(16, int(np.ceil(args.num_disparities / 16.0) * 16))
    block = args.block_size if args.block_size % 2 == 1 else args.block_size + 1
    block = max(3, block)
    return cv2.StereoSGBM_create(
        minDisparity=args.min_disparity,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=args.uniqueness_ratio,
        speckleWindowSize=args.speckle_window_size,
        speckleRange=args.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(cv2, matcher, left_rgb, right_rgb):
    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
    disp = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    invalid = ~np.isfinite(disp) | (disp <= 0.0)
    disp[invalid] = np.nan
    return disp


def disparity_stats(disp, prev_disp=None):
    valid = np.isfinite(disp)
    stats = {
        "valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "disp_mean": None,
        "disp_std": None,
        "disp_p05": None,
        "disp_p95": None,
        "temporal_absdiff_mean": None,
    }
    if valid.any():
        vals = disp[valid]
        stats.update(
            {
                "disp_mean": float(np.mean(vals)),
                "disp_std": float(np.std(vals)),
                "disp_p05": float(np.percentile(vals, 5)),
                "disp_p95": float(np.percentile(vals, 95)),
            }
        )
    if prev_disp is not None and prev_disp.shape == disp.shape:
        both = np.isfinite(disp) & np.isfinite(prev_disp)
        if both.any():
            stats["temporal_absdiff_mean"] = float(np.mean(np.abs(disp[both] - prev_disp[both])))
    return stats


def colorize_disparity(cv2, disp, vmin=None, vmax=None):
    valid = np.isfinite(disp)
    if not valid.any():
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    vals = disp[valid]
    if vmin is None:
        vmin = float(np.percentile(vals, 2))
    if vmax is None:
        vmax = float(np.percentile(vals, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-3
    norm = np.zeros(disp.shape, dtype=np.float32)
    norm[valid] = np.clip((disp[valid] - vmin) / (vmax - vmin), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~valid] = 0
    return color


def put_label(cv2, rgb, label):
    out = rgb.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (min(bgr.shape[1], 360), 30), (0, 0, 0), -1)
    cv2.putText(bgr, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stack_panel(cv2, parts):
    labelled = [put_label(cv2, image, label) for label, image in parts]
    h = min(p.shape[0] for p in labelled)
    resized = []
    for image in labelled:
        if image.shape[0] != h:
            w = int(round(image.shape[1] * (h / image.shape[0])))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        resized.append(image)
    return np.concatenate(resized, axis=1)


def write_png(cv2, path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def open_video_writer(cv2, path, fps, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def add_video_frame(cv2, writer, rgb):
    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def matched_frames(args):
    ed_left = index_images(args.ed3dgs_left_dir)
    ed_right = index_images(args.ed3dgs_root / "right")
    sc_left = index_images(args.stereocrafter_left_dir)
    sc_right = index_images(args.stereocrafter_right_dir)

    ed_frames = set(ed_left) & set(ed_right)
    sc_frames = set(sc_left) & set(sc_right)
    frames = sorted(ed_frames & sc_frames)
    if args.frame_start is not None:
        frames = [f for f in frames if f >= args.frame_start]
    if args.frame_end is not None:
        frames = [f for f in frames if f <= args.frame_end]
    if args.max_frames:
        frames = frames[: args.max_frames]
    return frames, {
        "ed3dgs": (ed_left, ed_right),
        "stereocrafter": (sc_left, sc_right),
        "counts": {
            "ed3dgs_left": len(ed_left),
            "ed3dgs_right": len(ed_right),
            "stereocrafter_left": len(sc_left),
            "stereocrafter_right": len(sc_right),
            "matched": len(frames),
        },
    }


def process_model(cv2, matcher, name, fid, left_path, right_path, out_dir, args, prev_disp):
    left = read_rgb(left_path)
    right = read_rgb(right_path)
    left, right = resize_pair(cv2, left, right, args.target_width, args.target_height)
    disp = compute_disparity(cv2, matcher, left, right)
    stats = disparity_stats(disp, prev_disp)
    if args.save_raw_npz:
        raw_path = out_dir / name / "disparity_raw" / f"{fid:06d}.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, disparity=disp.astype(np.float32))
    return left, right, disp, stats


def main():
    args = parse_args()
    cv2 = require_cv2()
    matcher = make_matcher(cv2, args)
    frames, sources = matched_frames(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise SystemExit(
            "No matched frame ids found.\n"
            f"E-D3DGS root: {args.ed3dgs_root}\n"
            f"StereoCrafter left: {args.stereocrafter_left_dir}\n"
            f"StereoCrafter right: {args.stereocrafter_right_dir}\n"
            f"Counts: {sources['counts']}"
        )

    metadata = {
        "ed3dgs_root": str(args.ed3dgs_root),
        "stereocrafter_left_dir": str(args.stereocrafter_left_dir),
        "stereocrafter_right_dir": str(args.stereocrafter_right_dir),
        "out_dir": str(args.out_dir),
        "frames": frames,
        "counts": sources["counts"],
        "sgbm": {
            "min_disparity": args.min_disparity,
            "num_disparities": int(np.ceil(args.num_disparities / 16.0) * 16),
            "block_size": args.block_size if args.block_size % 2 == 1 else args.block_size + 1,
            "target_width": args.target_width,
            "target_height": args.target_height,
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rows = []
    prev = {"ed3dgs": None, "stereocrafter": None}
    video_writers = {}

    try:
        for idx, fid in enumerate(frames):
            ed_left_idx, ed_right_idx = sources["ed3dgs"]
            sc_left_idx, sc_right_idx = sources["stereocrafter"]

            ed_left, ed_right, ed_disp, ed_stats = process_model(
                cv2, matcher, "ed3dgs", fid, ed_left_idx[fid], ed_right_idx[fid], args.out_dir, args, prev["ed3dgs"]
            )
            sc_left, sc_right, sc_disp, sc_stats = process_model(
                cv2, matcher, "stereocrafter", fid, sc_left_idx[fid], sc_right_idx[fid], args.out_dir, args, prev["stereocrafter"]
            )
            prev["ed3dgs"] = ed_disp
            prev["stereocrafter"] = sc_disp

            combined_valid = np.concatenate([ed_disp[np.isfinite(ed_disp)], sc_disp[np.isfinite(sc_disp)]])
            if combined_valid.size:
                vmin = float(np.percentile(combined_valid, 2))
                vmax = float(np.percentile(combined_valid, 98))
            else:
                vmin, vmax = 0.0, 1.0

            ed_vis = colorize_disparity(cv2, ed_disp, vmin, vmax)
            sc_vis = colorize_disparity(cv2, sc_disp, vmin, vmax)

            ed_panel = stack_panel(cv2, [("E-D3DGS left", ed_left), ("E-D3DGS right", ed_right), ("E-D3DGS disparity", ed_vis)])
            sc_panel = stack_panel(
                cv2,
                [("StereoCrafter source left", sc_left), ("StereoCrafter generated right", sc_right), ("StereoCrafter disparity", sc_vis)],
            )
            compare_panel = stack_panel(
                cv2,
                [
                    ("E-D3DGS left", ed_left),
                    ("E-D3DGS disparity", ed_vis),
                    ("StereoCrafter left", sc_left),
                    ("StereoCrafter disparity", sc_vis),
                ],
            )

            write_png(cv2, args.out_dir / "ed3dgs" / "disparity_vis" / f"{fid:06d}.png", ed_vis)
            write_png(cv2, args.out_dir / "stereocrafter" / "disparity_vis" / f"{fid:06d}.png", sc_vis)
            write_png(cv2, args.out_dir / "panels" / "ed3dgs" / f"{fid:06d}.png", ed_panel)
            write_png(cv2, args.out_dir / "panels" / "stereocrafter" / f"{fid:06d}.png", sc_panel)
            write_png(cv2, args.out_dir / "panels" / "comparison" / f"{fid:06d}.png", compare_panel)

            for key, panel in [("ed3dgs", ed_panel), ("stereocrafter", sc_panel), ("comparison", compare_panel)]:
                if key not in video_writers:
                    h, w = panel.shape[:2]
                    video_writers[key] = open_video_writer(cv2, args.out_dir / "videos" / f"{key}.mp4", args.fps, (w, h))
                add_video_frame(cv2, video_writers[key], panel)

            for model_name, stats in [("ed3dgs", ed_stats), ("stereocrafter", sc_stats)]:
                rows.append({"frame": fid, "model": model_name, **stats})

            if (idx + 1) % 25 == 0 or idx + 1 == len(frames):
                print(f"Processed {idx + 1}/{len(frames)} frames")
    finally:
        for writer in video_writers.values():
            writer.release()

    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        fieldnames = ["frame", "model", "valid_ratio", "disp_mean", "disp_std", "disp_p05", "disp_p95", "temporal_absdiff_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for model_name in ["ed3dgs", "stereocrafter"]:
        model_rows = [r for r in rows if r["model"] == model_name]
        summary[model_name] = {}
        for key in ["valid_ratio", "disp_mean", "disp_std", "temporal_absdiff_mean"]:
            vals = [r[key] for r in model_rows if r[key] is not None]
            summary[model_name][key] = float(np.mean(vals)) if vals else None
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
