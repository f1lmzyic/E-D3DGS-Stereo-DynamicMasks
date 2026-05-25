#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

from pathlib import Path
import os
import json
from argparse import ArgumentParser

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from tqdm import tqdm

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr


def _sorted_image_files(path):
    path = Path(path)
    exts = {".png", ".jpg", ".jpeg"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _default_device():
    return torch.device("cuda:0") if torch.cuda.is_available() and torch.cuda.device_count() > 0 else torch.device("cpu")


def readImages(renders_dir, gt_dir, device=None):
    if device is None:
        device = _default_device()
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        if not render_path.is_file() or not gt_path.is_file():
            continue
        render = Image.open(render_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        if render.size != gt.size:
            gt = gt.resize(render.size, Image.BILINEAR)

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].to(device))
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].to(device))
        image_names.append(fname)
    return renders, gts, image_names


def _tensor_to_uint8_img(x):
    arr = x.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def temporal_flickering_score(frames_uint8):
    """VBench-style temporal flickering score: (255 - mean adjacent MAE) / 255.

    Higher is better. 1.0 means no adjacent-frame change. This is a no-reference
    stability score, not an accuracy score against GT.
    """
    if len(frames_uint8) < 2:
        return None
    maes = []
    for a, b in zip(frames_uint8[:-1], frames_uint8[1:]):
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        maes.append(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    mean_mae = float(np.mean(maes))
    return float(np.clip((255.0 - mean_mae) / 255.0, 0.0, 1.0)), mean_mae


def imaging_quality_score(frames_uint8, device="cuda"):
    """VBench-style IQ score using MUSIQ from pyiqa when available.

    Returns mean(MUSIQ)/100. If pyiqa is not installed, returns (None, message).
    """
    try:
        import pyiqa
    except Exception as e:
        return None, f"pyiqa not available: {e}"

    try:
        metric = pyiqa.create_metric("musiq", device=device)
        vals = []
        with torch.no_grad():
            for img in frames_uint8:
                pil = Image.fromarray(img)
                # VBench commonly limits the longer side to 512 for IQ evaluation.
                w, h = pil.size
                longer = max(w, h)
                if longer > 512:
                    scale = 512.0 / longer
                    pil = pil.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
                t = tf.to_tensor(pil).unsqueeze(0).to(device)
                vals.append(float(metric(t).detach().cpu().item()))
        return float(np.mean(vals) / 100.0), None
    except Exception as e:
        return None, f"MUSIQ failed: {e}"


def _farneback_flow(prev_uint8, next_uint8):
    prev = cv2.cvtColor(prev_uint8, cv2.COLOR_RGB2GRAY) if prev_uint8.ndim == 3 else prev_uint8
    nxt = cv2.cvtColor(next_uint8, cv2.COLOR_RGB2GRAY) if next_uint8.ndim == 3 else next_uint8
    return cv2.calcOpticalFlowFarneback(
        prev, nxt, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    ).astype(np.float32)


def proxy_epe(render_frames_uint8, gt_frames_uint8):
    """Proxy EPE: endpoint error between Farneback flow(render) and flow(GT).

    This is not a replacement for true GT optical-flow EPE. It is useful when no
    GT flow exists and you want a temporal-motion fidelity signal.
    """
    n = min(len(render_frames_uint8), len(gt_frames_uint8))
    if n < 2:
        return None
    epes = []
    for i in range(n - 1):
        r0, r1 = render_frames_uint8[i], render_frames_uint8[i + 1]
        g0, g1 = gt_frames_uint8[i], gt_frames_uint8[i + 1]
        if g0.shape != r0.shape:
            g0 = cv2.resize(g0, (r0.shape[1], r0.shape[0]), interpolation=cv2.INTER_LINEAR)
        if g1.shape != r1.shape:
            g1 = cv2.resize(g1, (r1.shape[1], r1.shape[0]), interpolation=cv2.INTER_LINEAR)
        flow_r = _farneback_flow(r0, r1)
        flow_g = _farneback_flow(g0, g1)
        epe = np.sqrt(((flow_r - flow_g) ** 2).sum(axis=2))
        epes.append(float(epe.mean()))
    return float(np.mean(epes))


def _sgbm_disparity(left_uint8, right_uint8, max_disp=128):
    left = cv2.cvtColor(left_uint8, cv2.COLOR_RGB2GRAY) if left_uint8.ndim == 3 else left_uint8
    right = cv2.cvtColor(right_uint8, cv2.COLOR_RGB2GRAY) if right_uint8.ndim == 3 else right_uint8
    num_disp = int(np.ceil(max_disp / 16.0) * 16)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = matcher.compute(left, right).astype(np.float32) / 16.0
    disp[disp < 0] = np.nan
    return disp


def proxy_d1_all(stereo_method_dir):
    """Proxy D1-all from stereo render folders using SGBM disparities.

    Expects render_stereo.py output:
      renders/left, renders/right, gt, gt_right

    D1-all is the KITTI bad-pixel rate: |d_pred-d_gt| > 3 px and >5% of GT disp.
    Here d_pred and d_gt are both estimated by SGBM unless real GT disparity is
    supplied elsewhere, so treat this as a proxy stereo consistency metric.
    """
    stereo_method_dir = Path(stereo_method_dir)
    left_dir = stereo_method_dir / "renders" / "left"
    right_dir = stereo_method_dir / "renders" / "right"
    gt_left_dir = stereo_method_dir / "gt"
    gt_right_dir = stereo_method_dir / "gt_right"
    if not (left_dir.is_dir() and right_dir.is_dir() and gt_left_dir.is_dir() and gt_right_dir.is_dir()):
        return None

    files = _sorted_image_files(left_dir)
    if not files:
        return None
    vals = []
    for p in tqdm(files, desc="Proxy D1-all", leave=False):
        name = p.name
        paths = [left_dir / name, right_dir / name, gt_left_dir / name, gt_right_dir / name]
        if not all(x.is_file() for x in paths):
            continue
        l = np.array(Image.open(paths[0]).convert("RGB"))
        r = np.array(Image.open(paths[1]).convert("RGB"))
        gl = np.array(Image.open(paths[2]).convert("RGB"))
        gr = np.array(Image.open(paths[3]).convert("RGB"))
        if gl.shape[:2] != l.shape[:2]:
            gl = cv2.resize(gl, (l.shape[1], l.shape[0]), interpolation=cv2.INTER_LINEAR)
        if gr.shape[:2] != r.shape[:2]:
            gr = cv2.resize(gr, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_LINEAR)
        d_pred = _sgbm_disparity(l, r)
        d_gt = _sgbm_disparity(gl, gr)
        valid = np.isfinite(d_pred) & np.isfinite(d_gt) & (d_gt > 0)
        if valid.sum() == 0:
            continue
        err = np.abs(d_pred[valid] - d_gt[valid])
        bad = (err > 3.0) & ((err / np.abs(d_gt[valid])) > 0.05)
        vals.append(float(bad.mean() * 100.0))
    return float(np.mean(vals)) if vals else None


def _score_from_raw(metrics):
    """Normalize available metrics to [0,1] and average them into TotalScore.

    Higher raw is better: PSNR, SSIM, IQ-Score, TF-Score.
    Lower raw is better: LPIPS, EPE, D1-all.
    """
    scores = {}
    if metrics.get("PSNR") is not None:
        scores["PSNR_score"] = float(np.clip(metrics["PSNR"] / 40.0, 0.0, 1.0))
    if metrics.get("SSIM") is not None:
        scores["SSIM_score"] = float(np.clip(metrics["SSIM"], 0.0, 1.0))
    if metrics.get("LPIPS_ALEX") is not None:
        scores["LPIPS_score"] = float(np.clip(1.0 - metrics["LPIPS_ALEX"], 0.0, 1.0))
    elif metrics.get("LPIPS_VGG") is not None:
        scores["LPIPS_score"] = float(np.clip(1.0 - metrics["LPIPS_VGG"], 0.0, 1.0))
    if metrics.get("IQ-Score") is not None:
        scores["IQ_score_norm"] = float(np.clip(metrics["IQ-Score"], 0.0, 1.0))
    if metrics.get("TF-Score") is not None:
        scores["TF_score_norm"] = float(np.clip(metrics["TF-Score"], 0.0, 1.0))
    if metrics.get("EPE") is not None:
        scores["EPE_score"] = float(np.exp(-metrics["EPE"] / 10.0))
    if metrics.get("D1-all") is not None:
        scores["D1_all_score"] = float(np.clip(1.0 - metrics["D1-all"] / 100.0, 0.0, 1.0))
    scores["TotalScore"] = float(np.mean(list(scores.values()))) if scores else None
    return scores


def _find_stereo_method_dir(scene_dir, split_name, method):
    split_dir = Path(scene_dir) / split_name
    if not split_dir.is_dir():
        return None
    # Prefer exact stereo method name if metrics are being run on stereo_*.
    exact = split_dir / method
    if exact.is_dir() and (exact / "renders" / "left").is_dir():
        return exact
    stereo_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("stereo_")])
    return stereo_dirs[-1] if stereo_dirs else None


def _compute_image_sequence_metrics(renders_dir, gt_dir, compute_iq=False, compute_tf=True, compute_proxy_epe=True, label="", device=None):
    if device is None:
        device = _default_device()
    renders, gts, image_names = readImages(Path(renders_dir), Path(gt_dir), device=device)
    if len(renders) == 0:
        return None, None

    ssims = []
    psnrs = []
    lpips_vggs = []
    lpips_alexs = []

    desc = f"Metric evaluation {label}" if label else "Metric evaluation progress"
    for idx in tqdm(range(len(renders)), desc=desc):
        ssims.append(ssim(renders[idx], gts[idx])[0])
        psnrs.append(psnr(renders[idx], gts[idx]))
        lpips_vggs.append(lpips(renders[idx], gts[idx], net_type="vgg"))
        lpips_alexs.append(lpips(renders[idx], gts[idx], net_type="alex"))

    render_frames = [_tensor_to_uint8_img(x) for x in renders]
    gt_frames = [_tensor_to_uint8_img(x) for x in gts]

    metrics = {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS_VGG": torch.tensor(lpips_vggs).mean().item(),
        "LPIPS_ALEX": torch.tensor(lpips_alexs).mean().item(),
    }

    if compute_iq:
        iq, iq_note = imaging_quality_score(render_frames, device=str(device))
        metrics["IQ-Score"] = iq
        if iq_note:
            metrics["IQ-Score_note"] = iq_note
    else:
        metrics["IQ-Score"] = None
        metrics["IQ-Score_note"] = "Disabled. Re-run with --compute_iq and install pyiqa for VBench MUSIQ IQ-Score."

    if compute_tf:
        tf_render = temporal_flickering_score(render_frames)
        tf_gt = temporal_flickering_score(gt_frames)
        if tf_render is not None:
            metrics["TF-Score"] = tf_render[0]
            metrics["TF-MAE"] = tf_render[1]
        if tf_gt is not None:
            metrics["GT_TF-Score"] = tf_gt[0]
            metrics["GT_TF-MAE"] = tf_gt[1]

    if compute_proxy_epe:
        metrics["EPE"] = proxy_epe(render_frames, gt_frames)
        metrics["EPE_note"] = "Proxy EPE: Farneback flow(render frames) vs Farneback flow(GT frames), not true GT optical-flow EPE."

    per_view = {
        "SSIM": {name: val for val, name in zip(torch.tensor(ssims).tolist(), image_names)},
        "PSNR": {name: val for val, name in zip(torch.tensor(psnrs).tolist(), image_names)},
        "LPIPS_VGG": {name: val for val, name in zip(torch.tensor(lpips_vggs).tolist(), image_names)},
        "LPIPS_ALEX": {name: val for val, name in zip(torch.tensor(lpips_alexs).tolist(), image_names)},
    }
    return metrics, per_view


def _average_left_right(left_metrics, right_metrics):
    avg = {}
    for key in ["SSIM", "PSNR", "LPIPS_VGG", "LPIPS_ALEX", "IQ-Score", "TF-Score", "TF-MAE", "GT_TF-Score", "GT_TF-MAE", "EPE"]:
        vals = []
        for m in [left_metrics, right_metrics]:
            if m is not None and m.get(key) is not None:
                vals.append(m[key])
        if vals:
            avg[key] = float(np.mean(vals))
    if left_metrics and left_metrics.get("IQ-Score_note"):
        avg["IQ-Score_note"] = left_metrics["IQ-Score_note"]
    if left_metrics and left_metrics.get("EPE_note"):
        avg["EPE_note"] = left_metrics["EPE_note"]
    return avg


def _print_metric_summary(scene_dir, key, metrics):
    print("Scene:", scene_dir, "Split/Method:", key)
    for k in ["PSNR", "SSIM", "LPIPS_ALEX", "LPIPS_VGG", "IQ-Score", "TF-Score", "EPE", "D1-all", "TotalScore"]:
        if k in metrics and metrics[k] is not None:
            print(f"  {k}: {metrics[k]:.7f}")
        elif k in metrics:
            print(f"  {k}: unavailable")
    print("")


def _is_stereo_method_dir(method_dir):
    method_dir = Path(method_dir)
    return (
        (method_dir / "renders" / "left").is_dir()
        and (method_dir / "renders" / "right").is_dir()
        and (method_dir / "gt").is_dir()
        and (method_dir / "gt_right").is_dir()
    )


def evaluate(model_paths, test_paths=None, compute_iq=False, compute_tf=True, compute_proxy_epe=True, compute_proxy_d1=True, device=None):
    if device is None:
        device = _default_device()
    print(f"Metric device: {device}")
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        split_candidates = ["test", "train"]
        split_dirs = [Path(scene_dir) / s for s in split_candidates if (Path(scene_dir) / s).is_dir()]
        if not split_dirs:
            print(f"No train/test render folders found under {scene_dir}")
            continue

        for split_dir in split_dirs:
            split_name = split_dir.name
            for method in sorted(os.listdir(split_dir)):
                method_dir = split_dir / method
                if not method_dir.is_dir():
                    continue

                # Stereo render output: compute left, right, and left/right average.
                if _is_stereo_method_dir(method_dir):
                    print("Split:", split_name, "Stereo Method:", method)
                    left_metrics, left_per_view = _compute_image_sequence_metrics(
                        method_dir / "renders" / "left",
                        method_dir / "gt",
                        compute_iq=compute_iq,
                        compute_tf=compute_tf,
                        compute_proxy_epe=compute_proxy_epe,
                        label=f"{split_name}/{method}/left",
                        device=device,
                    )
                    right_metrics, right_per_view = _compute_image_sequence_metrics(
                        method_dir / "renders" / "right",
                        method_dir / "gt_right",
                        compute_iq=compute_iq,
                        compute_tf=compute_tf,
                        compute_proxy_epe=compute_proxy_epe,
                        label=f"{split_name}/{method}/right",
                        device=device,
                    )
                    if left_metrics is None and right_metrics is None:
                        continue

                    d1 = proxy_d1_all(method_dir) if compute_proxy_d1 else None
                    d1_note = "Proxy D1-all from SGBM(render L/R) vs SGBM(GT L/R). Requires render_stereo.py outputs with gt_right."

                    if left_metrics is not None:
                        left_metrics["D1-all"] = d1
                        left_metrics["D1-all_note"] = d1_note
                        left_metrics.update(_score_from_raw(left_metrics))
                        key = f"{split_name}/{method}/left"
                        full_dict[scene_dir][key] = left_metrics
                        per_view_dict[scene_dir][key] = left_per_view
                        _print_metric_summary(scene_dir, key, left_metrics)

                    if right_metrics is not None:
                        right_metrics["D1-all"] = d1
                        right_metrics["D1-all_note"] = d1_note
                        right_metrics.update(_score_from_raw(right_metrics))
                        key = f"{split_name}/{method}/right"
                        full_dict[scene_dir][key] = right_metrics
                        per_view_dict[scene_dir][key] = right_per_view
                        _print_metric_summary(scene_dir, key, right_metrics)

                    avg_metrics = _average_left_right(left_metrics, right_metrics)
                    avg_metrics["D1-all"] = d1
                    avg_metrics["D1-all_note"] = d1_note
                    avg_metrics.update(_score_from_raw(avg_metrics))
                    key = f"{split_name}/{method}/avg_left_right"
                    full_dict[scene_dir][key] = avg_metrics
                    per_view_dict[scene_dir][key] = {"note": "Average of aggregate left and right metrics; no per-frame values."}
                    _print_metric_summary(scene_dir, key, avg_metrics)
                    continue

                # Mono render output.
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                if not (gt_dir.is_dir() and renders_dir.is_dir()):
                    continue
                if len(_sorted_image_files(renders_dir)) == 0:
                    continue

                print("Split:", split_name, "Method:", method)
                metrics, per_view = _compute_image_sequence_metrics(
                    renders_dir,
                    gt_dir,
                    compute_iq=compute_iq,
                    compute_tf=compute_tf,
                    compute_proxy_epe=compute_proxy_epe,
                    label=f"{split_name}/{method}",
                    device=device,
                )
                if metrics is None:
                    continue

                if compute_proxy_d1:
                    stereo_method_dir = _find_stereo_method_dir(scene_dir, split_name, method)
                    metrics["D1-all"] = proxy_d1_all(stereo_method_dir) if stereo_method_dir is not None else None
                    metrics["D1-all_note"] = "Proxy D1-all from SGBM(render L/R) vs SGBM(GT L/R). Requires render_stereo.py outputs with gt_right."

                metrics.update(_score_from_raw(metrics))
                key = f"{split_name}/{method}"
                full_dict[scene_dir][key] = metrics
                per_view_dict[scene_dir][key] = per_view
                _print_metric_summary(scene_dir, key, metrics)

        with open(os.path.join(scene_dir, "results.json"), "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=2)
        with open(os.path.join(scene_dir, "per_view.json"), "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=2)


if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(torch.device("cuda:0"))

    parser = ArgumentParser(description="Metric evaluation script")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    parser.add_argument("--test_paths", type=str, default=[])
    parser.add_argument("--compute_iq", action="store_true", help="Compute VBench-style MUSIQ IQ-Score via pyiqa")
    parser.add_argument("--no_tf", action="store_true", help="Disable VBench-style temporal flickering score")
    parser.add_argument("--no_proxy_epe", action="store_true", help="Disable proxy optical-flow EPE")
    parser.add_argument("--no_proxy_d1", action="store_true", help="Disable proxy D1-all stereo metric")
    parser.add_argument("--device", default=None, help="Metric device, e.g. cuda:0 or cpu. Defaults to cuda if visible, else cpu.")

    args = parser.parse_args()
    evaluate(
        args.model_paths,
        args.test_paths,
        compute_iq=args.compute_iq,
        compute_tf=not args.no_tf,
        compute_proxy_epe=not args.no_proxy_epe,
        compute_proxy_d1=not args.no_proxy_d1,
        device=torch.device(args.device) if args.device is not None else None,
    )
