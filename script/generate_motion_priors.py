#!/usr/bin/env python3
"""Generate camera-compensated residual motion-prior sidecars.

The output is intentionally a soft 2D cue, not a segmentation mask.  It is meant
for conservative train-time weighting/oversampling/densification in train.py.
Each image gets a matching .npz sidecar under --output with keys:
  - confidence: HxW float16 in [0, 1], compact residual-motion confidence
  - residual_mag: HxW float16, residual optical-flow magnitude in pixels
  - flow: HxWx2 float16, original frame-to-next optical flow
  - camera_flow: HxWx2 float16, fitted global camera/parallax flow

No colour-specific object detector is used.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def list_images(images_root):
    exts = {".png", ".jpg", ".jpeg"}
    by_view = {}
    for p in sorted(Path(images_root).rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(images_root)
        view = str(rel.parent)
        by_view.setdefault(view, []).append(p)
    return by_view


def read_rgb(path, max_size=0):
    img = Image.open(path).convert("RGB")
    orig_size = img.size
    scale = 1.0
    if max_size and max(orig_size) > max_size:
        scale = float(max_size) / float(max(orig_size))
        img = img.resize((max(1, int(round(orig_size[0] * scale))), max(1, int(round(orig_size[1] * scale)))), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    return arr, orig_size, scale


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def compute_flow_farneback(img0, img1):
    cv2 = _try_import_cv2()
    if cv2 is None:
        return None
    g0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        g0, g1, None,
        pyr_scale=0.5, levels=5, winsize=21, iterations=5,
        poly_n=7, poly_sigma=1.5, flags=0,
    ).astype(np.float32)


def estimate_translation_fft(img0, img1):
    """Estimate integer global translation img0->img1 using phase correlation."""
    g0 = img0.astype(np.float32).mean(axis=2)
    g1 = img1.astype(np.float32).mean(axis=2)
    g0 = g0 - g0.mean()
    g1 = g1 - g1.mean()
    cps = np.fft.fft2(g0) * np.conj(np.fft.fft2(g1))
    cps /= np.maximum(np.abs(cps), 1e-6)
    corr = np.fft.ifft2(cps).real
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = g0.shape
    if x > w // 2:
        x -= w
    if y > h // 2:
        y -= h
    # np.roll(img1, (y, x)) aligns img1 back to img0, so flow img0->img1 is (-x, -y).
    return float(-x), float(-y)


def residual_from_aligned_difference(img0, img1):
    """Fallback when dense optical flow is unavailable: align global camera shift and diff."""
    dx, dy = estimate_translation_fft(img0, img1)
    aligned1 = np.roll(img1, shift=(int(round(-dy)), int(round(-dx))), axis=(0, 1))
    diff = np.abs(img0.astype(np.float32) - aligned1.astype(np.float32)).mean(axis=2)
    h, w = diff.shape
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[..., 0] = dx
    flow[..., 1] = dy
    cam_flow = flow.copy()
    return diff.astype(np.float32), flow, cam_flow


_RAFT_CACHE = {}


def compute_flow_raft_torchvision(img0, img1, device="cuda", torch_home=None):
    if torch_home:
        os.environ["TORCH_HOME"] = str(torch_home)
    import torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    key = (device, os.environ.get("TORCH_HOME", ""))
    if key not in _RAFT_CACHE:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True).to(device).eval()
        _RAFT_CACHE[key] = (model, weights.transforms())
    model, transforms = _RAFT_CACHE[key]

    t0 = torch.from_numpy(img0).permute(2, 0, 1).float()[None] / 255.0
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None] / 255.0
    h, w = t0.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        t0 = F.pad(t0, (0, pw, 0, ph), mode="replicate")
        t1 = F.pad(t1, (0, pw, 0, ph), mode="replicate")
    t0, t1 = transforms(t0, t1)
    with torch.no_grad():
        flow = model(t0.to(device), t1.to(device))[-1][0, :, :h, :w].permute(1, 2, 0).cpu().numpy()
    return flow.astype(np.float32)


def fit_camera_flow(flow, stride=8, ransac_thresh=2.0):
    """Fit a global affine flow with RANSAC and return dense fitted flow."""
    cv2 = _try_import_cv2()
    if cv2 is None:
        median = np.nanmedian(flow.reshape(-1, 2), axis=0)
        return np.broadcast_to(median.reshape(1, 1, 2), flow.shape).astype(np.float32)
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    src = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    f = flow[yy, xx].reshape(-1, 2).astype(np.float32)
    dst = src + f
    valid = np.isfinite(dst).all(axis=1) & np.isfinite(src).all(axis=1)
    src = src[valid]
    dst = dst[valid]
    if len(src) < 16:
        median = np.nanmedian(flow.reshape(-1, 2), axis=0)
        return np.broadcast_to(median.reshape(1, 1, 2), flow.shape).astype(np.float32)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=2000, confidence=0.99)
    if M is None:
        median = np.nanmedian(flow.reshape(-1, 2), axis=0)
        return np.broadcast_to(median.reshape(1, 1, 2), flow.shape).astype(np.float32)
    yy_full, xx_full = np.mgrid[0:h, 0:w]
    ones = np.ones_like(xx_full, dtype=np.float32)
    pts = np.stack([xx_full, yy_full, ones], axis=-1).astype(np.float32)
    dst_full = pts @ M.T
    cam_flow = dst_full - pts[..., :2]
    return cam_flow.astype(np.float32)


def resize_flow(flow, orig_size, scale):
    if scale == 1.0:
        return flow.astype(np.float32)
    ow, oh = orig_size
    cv2 = _try_import_cv2()
    if cv2 is not None:
        out = cv2.resize(flow, (ow, oh), interpolation=cv2.INTER_LINEAR)
    else:
        chans = []
        for c in range(flow.shape[-1]):
            chan = Image.fromarray(flow[..., c].astype(np.float32), mode="F").resize((ow, oh), Image.BILINEAR)
            chans.append(np.asarray(chan, dtype=np.float32))
        out = np.stack(chans, axis=-1)
    out[..., 0] /= scale
    out[..., 1] /= scale
    return out.astype(np.float32)


def filter_components(conf, threshold, min_area_px, max_area_frac, max_components, dilate=1):
    cv2 = _try_import_cv2()
    h, w = conf.shape
    mask = (conf >= threshold).astype(np.uint8)
    if cv2 is not None:
        if dilate > 0:
            k = max(1, int(dilate)) * 2 + 1
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = {i: int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)}
    else:
        try:
            from scipy import ndimage as ndi
            if dilate > 0:
                mask = ndi.binary_closing(mask, iterations=max(1, int(dilate))).astype(np.uint8)
            labels, n = ndi.label(mask)
            areas = {i: int((labels == i).sum()) for i in range(1, n + 1)}
        except Exception:
            return conf * mask.astype(np.float32)
    comps = []
    max_area_px = int(max_area_frac * h * w) if max_area_frac > 0 and max_area_frac <= 1 else int(max_area_frac)
    if max_area_px <= 0:
        max_area_px = h * w
    for i, area in areas.items():
        if area < min_area_px or area > max_area_px:
            continue
        score = float(conf[labels == i].mean())
        comps.append((score, i, area))
    comps.sort(reverse=True)
    keep = np.zeros_like(conf, dtype=np.float32)
    for _, i, _ in comps[:max_components]:
        keep[labels == i] = conf[labels == i]
    return keep


def save_overlay(image_path, conf, out_path):
    img = Image.open(image_path).convert("RGB")
    if conf.shape[::-1] != img.size:
        conf_img = Image.fromarray((np.clip(conf, 0, 1) * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
        conf = np.asarray(conf_img, dtype=np.float32) / 255.0
    overlay = img.copy().convert("RGBA")
    red = Image.new("RGBA", img.size, (255, 0, 0, 0))
    alpha = Image.fromarray((np.clip(conf, 0, 1) * 180).astype(np.uint8))
    red.putalpha(alpha)
    overlay.alpha_composite(red)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--images", default="images")
    p.add_argument("--output", default="motion_priors")
    p.add_argument("--overlay_dir", default="")
    p.add_argument("--method", choices=["farneback", "raft_torchvision"], default="farneback")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_home", default=os.environ.get("TORCH_HOME", ""), help="Torch hub cache root. For torchvision RAFT, checkpoint should be at <torch_home>/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth")
    p.add_argument("--max_size", type=int, default=768, help="flow processing max image side; 0 = full resolution")
    p.add_argument("--percentile", type=float, default=98.5)
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--min_area", type=int, default=4)
    p.add_argument("--max_area", type=float, default=0.05, help="component max area fraction if <=1, else pixels")
    p.add_argument("--max_components", type=int, default=8)
    p.add_argument("--camera_stride", type=int, default=8)
    p.add_argument("--ransac_thresh", type=float, default=2.0)
    p.add_argument("--component_dilate", type=int, default=1)
    args = p.parse_args()

    source = Path(args.source)
    images_root = source / args.images
    output_root = source / args.output
    overlay_root = source / args.overlay_dir if args.overlay_dir else None
    by_view = list_images(images_root)
    if not by_view:
        raise SystemExit(f"No images found under {images_root}")

    for view, paths in by_view.items():
        print(f"[motion-priors] {view}: {len(paths)} images")
        prev_conf = None
        prev_payload = None
        for idx, path0 in enumerate(paths):
            if idx < len(paths) - 1:
                path1 = paths[idx + 1]
                img0, orig_size, scale = read_rgb(path0, args.max_size)
                img1, _, _ = read_rgb(path1, args.max_size)
                if args.method == "raft_torchvision":
                    flow = compute_flow_raft_torchvision(img0, img1, device=args.device, torch_home=args.torch_home)
                else:
                    flow = compute_flow_farneback(img0, img1)
                if flow is None:
                    residual_mag, flow, cam_flow = residual_from_aligned_difference(img0, img1)
                    flow = resize_flow(flow, orig_size, scale)
                    cam_flow = resize_flow(cam_flow, orig_size, scale)
                    if scale != 1.0:
                        from PIL import Image as _Image
                        residual_mag = np.asarray(_Image.fromarray(residual_mag).resize(orig_size, _Image.BILINEAR), dtype=np.float32)
                else:
                    cam_flow = fit_camera_flow(flow, stride=args.camera_stride, ransac_thresh=args.ransac_thresh)
                    flow = resize_flow(flow, orig_size, scale)
                    cam_flow = resize_flow(cam_flow, orig_size, scale)
                    residual = flow - cam_flow
                    residual_mag = np.linalg.norm(residual, axis=-1).astype(np.float32)
                finite = np.isfinite(residual_mag)
                norm = max(float(np.percentile(residual_mag[finite], args.percentile)), 1e-6) if finite.any() else 1.0
                conf = np.clip(residual_mag / norm, 0.0, 1.0).astype(np.float32)
                conf = filter_components(conf, args.threshold, args.min_area, args.max_area, args.max_components, dilate=args.component_dilate)
                payload = dict(
                    confidence=conf.astype(np.float16),
                    residual_mag=residual_mag.astype(np.float16),
                    flow=flow.astype(np.float16),
                    camera_flow=cam_flow.astype(np.float16),
                )
                prev_conf, prev_payload = conf, payload
            else:
                # Last frame has no forward neighbour; reuse previous confidence if available.
                if prev_payload is None:
                    img0, orig_size, _ = read_rgb(path0, args.max_size)
                    h, w = img0.shape[:2]
                    zero_flow = np.zeros((h, w, 2), dtype=np.float16)
                    payload = dict(confidence=np.zeros((h, w), dtype=np.float16), residual_mag=np.zeros((h, w), dtype=np.float16), flow=zero_flow, camera_flow=zero_flow)
                    conf = np.zeros((h, w), dtype=np.float32)
                else:
                    payload = prev_payload
                    conf = prev_conf

            rel = path0.relative_to(images_root)
            out_npz = output_root / rel.with_suffix(".npz")
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_npz, **payload)
            # Also write an 8-bit PNG for quick inspection and compatibility.
            out_png = output_root / rel.with_suffix(".png")
            Image.fromarray((np.clip(payload["confidence"].astype(np.float32), 0, 1) * 255).astype(np.uint8)).save(out_png)
            if overlay_root is not None:
                save_overlay(path0, conf, overlay_root / rel)

    print(f"Motion priors written to: {output_root}")


if __name__ == "__main__":
    main()
