#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import numpy as np
import random
import os
import torch
import torch.nn.functional as F
from PIL import Image
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
from utils.extra_utils import o3d_knn, weighted_l2_loss_v2, image_sampler, calculate_distances, sample_camera

# import lpips
from utils.scene_utils import render_training_image
from time import time
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


class SyntheticShiftCamera:
    pass


def make_synthetic_right_camera(viewpoint_cam, baseline):
    """Clone a training camera and translate its center along the camera-local +x axis."""
    right_cam = SyntheticShiftCamera()
    right_cam.__dict__.update(viewpoint_cam.__dict__)

    c2w = torch.inverse(viewpoint_cam.world_view_transform.cuda())
    c2w_shifted = c2w.clone()
    right_axis_world = c2w_shifted[0, :3]
    c2w_shifted[3, :3] = c2w_shifted[3, :3] + float(baseline) * right_axis_world

    right_cam.world_view_transform = torch.inverse(c2w_shifted)
    right_cam.full_proj_transform = (
        right_cam.world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.cuda().unsqueeze(0))
    ).squeeze(0)
    right_cam.camera_center = c2w_shifted[3, :3]
    return right_cam


def stereo_consistency_loss(left_image, right_image, left_depth, left_cam, right_cam,
                            occlusion_tolerance=0.01, min_valid_pixels=64):
    """Forward-warp left render/depth into the synthetic right view and compare visible pixels.

    The actual loss is evaluated on projected left pixels by sampling the right render at the
    projected coordinates. A detached z-buffer in right-pixel space removes source pixels that
    become occluded from the right-eye view; disoccluded right pixels naturally have no source.
    """
    if left_depth is None:
        return None
    if left_depth.dim() == 2:
        left_depth = left_depth.unsqueeze(0)
    left_depth = left_depth[:1].float()

    _, h, w = left_image.shape
    device = left_image.device
    dtype = torch.float32
    depth = left_depth.to(device=device, dtype=dtype)
    if depth.shape[-2:] != (h, w):
        depth = F.interpolate(depth[None], size=(h, w), mode="bilinear", align_corners=False)[0]

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = w / (2.0 * math.tan(float(left_cam.FoVx) * 0.5))
    fy = h / (2.0 * math.tan(float(left_cam.FoVy) * 0.5))
    cx = (w - 1.0) * 0.5
    cy = (h - 1.0) * 0.5

    z = depth[0]
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z
    pts_left = torch.stack((x_cam, y_cam, z, torch.ones_like(z)), dim=-1).view(-1, 4)

    world = pts_left @ torch.inverse(left_cam.world_view_transform.to(device=device, dtype=dtype))
    right_view = world @ right_cam.world_view_transform.to(device=device, dtype=dtype)
    clip = world @ right_cam.full_proj_transform.to(device=device, dtype=dtype)
    clip_w = clip[:, 3:4]
    ndc = clip[:, :2] / clip_w.clamp_min(1e-8)
    z_right = right_view[:, 2]

    valid = (
        torch.isfinite(z.view(-1))
        & torch.isfinite(ndc).all(dim=-1)
        & (clip_w[:, 0] > 1e-8)
        & (z.view(-1) > 0)
        & (z_right > getattr(right_cam, "znear", 0.01))
        & (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    )
    if valid.sum() < min_valid_pixels:
        return None

    # Right-view z-buffer for projected left samples; keep only non-occluded source pixels.
    u = ((ndc[:, 0] + 1.0) * w - 1.0) * 0.5
    v = ((ndc[:, 1] + 1.0) * h - 1.0) * 0.5
    ui = u.round().long().clamp(0, w - 1)
    vi = v.round().long().clamp(0, h - 1)
    lin = vi * w + ui
    min_z = torch.full((h * w,), float("inf"), device=device, dtype=dtype)
    min_z.scatter_reduce_(0, lin[valid], z_right.detach()[valid], reduce="amin", include_self=True)
    zbuf = min_z[lin]
    non_occluded = valid & (z_right.detach() <= zbuf + float(occlusion_tolerance) * zbuf.clamp_min(1.0))
    if non_occluded.sum() < min_valid_pixels:
        return None

    sample_grid = ndc.view(1, h, w, 2)
    sampled_right = F.grid_sample(
        right_image.unsqueeze(0), sample_grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )[0]
    per_pixel = torch.abs(sampled_right - left_image).mean(dim=0).view(-1)
    return per_pixel[non_occluded].mean()


def _resize_1chw(x, hw, mode="bilinear"):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x[:1].float()
    if x.shape[-2:] != hw:
        x = F.interpolate(x[None], size=hw, mode=mode, align_corners=False if mode == "bilinear" else None)[0]
    return x


def layered_rgb_separation_loss(bg_image, fg_image, gt_image, dynamic_mask, bg_color):
    """Train background and foreground layers on complementary dynamic-mask regions."""
    _, h, w = gt_image.shape
    dyn = _resize_1chw(dynamic_mask, (h, w)).to(gt_image.device).clamp(0.0, 1.0)
    bg_color_img = bg_color.to(gt_image.device).view(3, 1, 1)
    outside = 1.0 - dyn
    inside_area = dyn.mean().clamp_min(1e-4)
    outside_area = outside.mean().clamp_min(1e-4)
    bg_outside = (torch.abs(bg_image - gt_image).mean(dim=0, keepdim=True) * outside).mean() / outside_area
    fg_inside = (torch.abs(fg_image - gt_image).mean(dim=0, keepdim=True) * dyn).mean() / inside_area
    bg_leak_inside = (torch.abs(bg_image - bg_color_img).mean(dim=0, keepdim=True) * dyn).mean() / inside_area
    fg_leak_outside = (torch.abs(fg_image).mean(dim=0, keepdim=True) * outside).mean() / outside_area
    return bg_outside + fg_inside + 0.5 * (bg_leak_inside + fg_leak_outside)


def project_gaussian_layer_depth_class_loss(render_pkg, viewpoint_cam, gaussians, dynamic_mask, target_depth, render_depth,
                                            close_thresh=0.15, min_points=32, use_depth_gate=False):
    """Classify Gaussians as foreground/background using mask projection plus DA3 depth closeness.

    A Gaussian projected inside the dynamic mask is encouraged to foreground only when its
    camera-space depth is close to the DA3 foreground depth at that pixel. Gaussians outside
    the mask are encouraged to background. This avoids assigning background Gaussians behind
    the moving object to the foreground layer just because they project into the mask.
    """
    means = render_pkg.get("means3D_final", None)
    radii = render_pkg.get("radii", None)
    if means is None or radii is None or dynamic_mask is None:
        return None
    means = means.detach()
    radii = radii.detach()
    device = means.device
    h, w = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
    ones = torch.ones((means.shape[0], 1), device=device, dtype=means.dtype)
    homog = torch.cat([means, ones], dim=-1)
    view = homog @ viewpoint_cam.world_view_transform.to(device=device, dtype=means.dtype)
    clip = homog @ viewpoint_cam.full_proj_transform.to(device=device, dtype=means.dtype)
    clip_w = clip[:, 3:4]
    ndc = clip[:, :2] / clip_w.clamp_min(1e-8)
    z = view[:, 2:3]
    visible = (radii > 0) & (clip_w[:, 0] > 1e-8) & torch.isfinite(ndc).all(dim=-1) & \
              (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) & (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    if visible.sum() < min_points:
        return None

    grid = ndc.view(1, -1, 1, 2).float()
    mask_img = _resize_1chw(dynamic_mask, (h, w)).to(device).clamp(0.0, 1.0)
    mask_sample = F.grid_sample(mask_img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False).view(-1, 1)
    target_fg = (mask_sample > 0.2).float()

    if use_depth_gate and target_depth is not None and render_depth is not None:
        rd = _resize_1chw(render_depth, (h, w)).to(device)
        td = _resize_1chw(target_depth, (h, w)).to(device)
        valid_depth = torch.isfinite(rd) & torch.isfinite(td) & (rd > 0) & (td > 0)
        if valid_depth.sum() > 128:
            x = rd[valid_depth]
            y = td[valid_depth]
            x_mean = x.mean()
            y_mean = y.mean()
            var = ((x - x_mean) ** 2).mean().clamp_min(1e-6)
            scale = ((x - x_mean) * (y - y_mean)).mean() / var
            shift = y_mean - scale * x_mean
            depth_sample = F.grid_sample(td[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False).view(-1, 1)
            aligned_z = scale.detach() * z.float() + shift.detach()
            depth_scale = torch.median(torch.abs(y - y.median())).clamp_min(1e-3)
            close = torch.exp(-torch.abs(aligned_z - depth_sample) / (float(close_thresh) * depth_scale))
            target_fg = target_fg * close.detach().clamp(0.0, 1.0)

    target_fg = target_fg.detach()
    prob = gaussians.get_foreground_prob.to(device).float().clamp(1e-4, 1.0 - 1e-4)
    valid = visible & torch.isfinite(target_fg[:, 0])
    if valid.sum() < min_points:
        return None
    # Upweight rare foreground points so they are not drowned out by background points.
    pos_frac = target_fg[valid].mean().detach().clamp_min(1e-3)
    weight = torch.where(target_fg > 0.05, 0.5 / pos_frac, torch.ones_like(target_fg))
    bce = F.binary_cross_entropy(prob, target_fg, weight=weight, reduction="none")
    return bce[valid].mean()


def foreground_scale_regularizer(gaussians, max_scale=0.02):
    fg = gaussians.get_foreground_prob.float()
    if fg.numel() == 0:
        return None
    too_large = F.relu(gaussians.get_scaling.max(dim=1, keepdim=True).values - float(max_scale))
    return (fg.detach() * too_large).mean()


def _sidecar_area(path, threshold=0.5):
    try:
        if path.endswith(".npz") or path.endswith(".npy"):
            data = np.load(path)
            if isinstance(data, np.lib.npyio.NpzFile):
                if "confidence" in data:
                    arr = data["confidence"].astype(np.float32)
                elif "motion" in data:
                    arr = data["motion"].astype(np.float32)
                elif "residual_mag" in data:
                    arr = data["residual_mag"].astype(np.float32)
                    finite = np.isfinite(arr)
                    if finite.any():
                        arr = arr / max(float(np.percentile(arr[finite], 99.0)), 1e-6)
                else:
                    arr = data[list(data.keys())[0]].astype(np.float32)
            else:
                arr = data.astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 3:
                arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
            return float((arr >= threshold).mean())
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        return float((arr >= int(threshold * 255)).mean())
    except Exception:
        return 0.0


def find_dynamic_frame_candidates(train_cams, min_area=0.0001, threshold=0.5):
    """Return frame ids whose dynamic masks contain enough high-confidence pixels.

    This gives tiny moving objects more training visits without injecting new
    points or protecting noisy points, avoiding the dotted/smeared failure mode.
    """
    return find_sidecar_frame_candidates(train_cams, "dynamic_mask_path", min_area=min_area, threshold=threshold)


def find_sidecar_frame_candidates(train_cams, path_attr, min_area=0.0001, threshold=0.5):
    frame_scores = {}
    for cam in train_cams:
        path = getattr(cam, path_attr, None)
        if path is None or not os.path.exists(path):
            continue
        area = _sidecar_area(path, threshold=threshold)
        if area >= min_area:
            frame_no = int(cam.frame_no)
            frame_scores[frame_no] = max(frame_scores.get(frame_no, 0.0), area)
    return sorted(frame_scores.keys()), frame_scores


def component_normalized_dynamic_loss(abs_err, dynamic_masks, threshold=0.35, min_area=4,
                                      max_area=6000, max_components=16):
    """Average error per connected dynamic component instead of per image pixel.

    Tiny objects can be tens of pixels in a 640x360 training image. A normal
    full-frame mean makes their gradients vanish. This term labels each mask,
    computes the mean error for every accepted component, then averages those
    component means so a tiny ball has comparable weight to a large blob.
    """
    losses = []
    b = abs_err.shape[0]
    if ndi is None:
        valid = dynamic_masks > float(threshold)
        if valid.any():
            return abs_err[valid.expand_as(abs_err)].mean()
        return None
    for i in range(b):
        m = dynamic_masks[i, 0].detach().float()
        labels_np, nlab = ndi.label((m > float(threshold)).cpu().numpy(), structure=np.ones((3, 3), dtype=np.uint8))
        if nlab == 0:
            continue
        labels_t = torch.from_numpy(labels_np).to(abs_err.device)
        areas = np.bincount(labels_np.reshape(-1))
        comps = []
        for lab in range(1, len(areas)):
            area = int(areas[lab])
            if area < int(min_area):
                continue
            if int(max_area) > 0 and area > int(max_area):
                continue
            comps.append((area, lab))
        comps.sort(reverse=True, key=lambda x: x[0])
        if int(max_components) > 0:
            comps = comps[:int(max_components)]
        for _, lab in comps:
            comp = labels_t == lab
            if comp.any():
                losses.append(abs_err[i, 0][comp].mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def _mask_components(mask, threshold=0.35, min_area=4, max_area=4000, max_components=16):
    if mask is None or ndi is None:
        return []
    m = mask.detach().float()
    if m.dim() == 3:
        m = m[0]
    labels, nlab = ndi.label((m > float(threshold)).cpu().numpy(), structure=np.ones((3, 3), dtype=np.uint8))
    if nlab == 0:
        return []
    areas = np.bincount(labels.reshape(-1))
    objs = ndi.find_objects(labels)
    comps = []
    for lab, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        area = int(areas[lab])
        if area < int(min_area):
            continue
        if int(max_area) > 0 and area > int(max_area):
            continue
        sy, sx = slc
        ys, xs = np.where(labels[sy, sx] == lab)
        if xs.size == 0:
            continue
        xs = xs + sx.start
        ys = ys + sy.start
        comps.append({
            "area": area,
            "label": lab,
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
            "pixels": np.stack([xs, ys], axis=1),
        })
    comps.sort(reverse=True, key=lambda c: c["area"])
    if int(max_components) > 0:
        comps = comps[:int(max_components)]
    return comps


def _pixel_to_world_ray(cam, x, y, h, w, device):
    z = torch.ones_like(x, device=device, dtype=torch.float32)
    x_cam = (2.0 * (x + 0.5) / float(w) - 1.0) * math.tan(float(cam.FoVx) * 0.5) * z
    y_cam = (1.0 - 2.0 * (y + 0.5) / float(h)) * math.tan(float(cam.FoVy) * 0.5) * z
    pts_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=1)
    c2w = torch.inverse(cam.world_view_transform.to(device))
    pts_world = pts_cam @ c2w
    pts_world = pts_world[:, :3] / pts_world[:, 3:4].clamp_min(1e-6)
    origin = cam.camera_center.to(device).float().view(1, 3).expand_as(pts_world)
    dirs = F.normalize(pts_world - origin, dim=1)
    return origin, dirs


def _closest_points_between_rays(o1, d1, o2, d2):
    # Batched least-squares closest point between two 3D rays; returns midpoint.
    w0 = o1 - o2
    a = (d1 * d1).sum(dim=1)
    b = (d1 * d2).sum(dim=1)
    c = (d2 * d2).sum(dim=1)
    d = (d1 * w0).sum(dim=1)
    e = (d2 * w0).sum(dim=1)
    denom = (a * c - b * b).clamp_min(1e-6)
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    p1 = o1 + s[:, None] * d1
    p2 = o2 + t[:, None] * d2
    return 0.5 * (p1 + p2)


def sample_stereo_mask_seed_points(left_cam, right_cam, points_per_component=8, threshold=0.35,
                                   min_area=4, max_area=4000, y_tolerance=12.0):
    """Triangulate seed points from matched left/right dynamic-mask components."""
    if left_cam is None or right_cam is None:
        return None, None
    if type(left_cam.original_image) == type(None):
        left_cam.load_image()
    if type(right_cam.original_image) == type(None):
        right_cam.load_image()
    if getattr(left_cam, "dynamic_mask", None) is None and getattr(left_cam, "dynamic_mask_path", None) is not None:
        left_cam.load_dynamic_mask()
    if getattr(right_cam, "dynamic_mask", None) is None and getattr(right_cam, "dynamic_mask_path", None) is not None:
        right_cam.load_dynamic_mask()
    lcomps = _mask_components(left_cam.dynamic_mask, threshold, min_area, max_area)
    rcomps = _mask_components(right_cam.dynamic_mask, threshold, min_area, max_area)
    if not lcomps or not rcomps:
        return None, None
    device = left_cam.original_image.device if left_cam.original_image.is_cuda else torch.device("cuda")
    h, w = int(left_cam.image_height), int(left_cam.image_width)
    xyzs, rgbs = [], []
    used_r = set()
    for lc in lcomps:
        candidates = [(abs(lc["cy"] - rc["cy"]), j, rc) for j, rc in enumerate(rcomps) if j not in used_r]
        candidates = [c for c in candidates if c[0] <= float(y_tolerance)]
        if not candidates:
            continue
        _, rj, rc = min(candidates, key=lambda x: x[0])
        used_r.add(rj)
        pix = lc["pixels"]
        n = min(int(points_per_component), pix.shape[0])
        if n <= 0:
            continue
        choice = np.random.choice(pix.shape[0], size=n, replace=False)
        lxy = pix[choice].astype(np.float32)
        # Preserve local object shape by applying the left component pixel offset
        # to the matched right component centroid.
        rxy = np.empty_like(lxy)
        rxy[:, 0] = float(rc["cx"]) + (lxy[:, 0] - float(lc["cx"]))
        rxy[:, 1] = float(rc["cy"]) + (lxy[:, 1] - float(lc["cy"]))
        rxy[:, 0] = np.clip(rxy[:, 0], 0, w - 1)
        rxy[:, 1] = np.clip(rxy[:, 1], 0, h - 1)
        lx = torch.from_numpy(lxy[:, 0]).to(device)
        ly = torch.from_numpy(lxy[:, 1]).to(device)
        rx = torch.from_numpy(rxy[:, 0]).to(device)
        ry = torch.from_numpy(rxy[:, 1]).to(device)
        lo, ld = _pixel_to_world_ray(left_cam, lx, ly, h, w, device)
        ro, rd = _pixel_to_world_ray(right_cam, rx, ry, h, w, device)
        pts = _closest_points_between_rays(lo, ld, ro, rd)
        valid = torch.isfinite(pts).all(dim=1)
        if valid.any():
            pts = pts[valid]
            left_img = left_cam.original_image.to(device)
            colors = left_img[:, ly.long()[valid], lx.long()[valid]].permute(1, 0).contiguous()
            xyzs.append(pts)
            rgbs.append(colors)
    if not xyzs:
        return None, None
    return torch.cat(xyzs, dim=0), torch.cat(rgbs, dim=0)


def project_gaussians_into_mask(render_pkg, viewpoint_cam, dynamic_mask, threshold=0.25):
    """Return a bool mask over Gaussians whose projected centers land in the dynamic mask.

    This is used only to bias densification. A small fast object can be correctly
    masked in image space but still disappear if normal gradient-based densification
    spends almost all new points on the high-gradient static background. Boosting
    the screen-space densification gradient for Gaussians inside the dynamic mask
    encourages clones/splits around the object without changing the RGB objective.
    """
    means = render_pkg.get("means3D_final", None)
    radii = render_pkg.get("radii", None)
    if means is None or radii is None or dynamic_mask is None:
        return None
    with torch.no_grad():
        device = means.device
        n = means.shape[0]
        h, w = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
        mask_img = _resize_1chw(dynamic_mask, (h, w)).to(device).float().clamp(0.0, 1.0)
        pts_h = torch.cat([means.detach(), torch.ones((n, 1), device=device, dtype=means.dtype)], dim=-1)
        clip = pts_h @ viewpoint_cam.full_proj_transform.to(device)
        denom = clip[:, 3]
        valid = denom.abs() > 1e-6
        safe_denom = torch.where(denom.abs() > 1e-6, denom, torch.ones_like(denom))
        ndc = clip[:, :3] / safe_denom[:, None]
        x = ((ndc[:, 0] + 1.0) * 0.5 * (w - 1)).round().long()
        y = ((1.0 - ndc[:, 1]) * 0.5 * (h - 1)).round().long()
        valid = valid & (radii.detach() > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        selected = torch.zeros((n,), device=device, dtype=torch.bool)
        if valid.any():
            selected[valid] = mask_img[0, y[valid], x[valid]] > float(threshold)
        return selected


def sample_mask_seed_points(viewpoint_cam, gt_image, dynamic_mask, render_depth, target_depth=None, points_per_frame=32,
                            threshold=0.35, seed_scale=0.95, default_depth=2.0):
    if dynamic_mask is None or points_per_frame <= 0:
        return None, None
    with torch.no_grad():
        device = gt_image.device
        _, h, w = gt_image.shape
        mask_img = _resize_1chw(dynamic_mask, (h, w)).to(device).float().clamp(0.0, 1.0)
        ys, xs = torch.where(mask_img[0] > float(threshold))
        if xs.numel() == 0:
            return None, None
        n = min(int(points_per_frame), xs.numel())
        choice = torch.randperm(xs.numel(), device=device)[:n]
        xs = xs[choice].float()
        ys = ys[choice].float()
        colors = gt_image[:, ys.long(), xs.long()].permute(1, 0).contiguous()
        depth = target_depth if target_depth is not None else render_depth
        using_relative_target_depth = target_depth is not None and render_depth is None
        if depth is None:
            z = torch.ones_like(xs) * float(default_depth)
        else:
            if depth.dim() == 3:
                depth = depth[0]
            if depth.shape[-2:] != (h, w):
                depth = F.interpolate(depth[None, None] if depth.dim() == 2 else depth[None], size=(h, w), mode="bilinear", align_corners=False).squeeze()
            z = depth[ys.long(), xs.long()].float().to(device)
            valid_z = z[torch.isfinite(z) & (z > 0)]
            if valid_z.numel() == 0:
                z = torch.ones_like(xs) * float(default_depth)
            else:
                fill = valid_z.median()
                z = torch.where(torch.isfinite(z) & (z > 0), z, fill)
                # DA3 monocular depth is relative, not in COLMAP scene units. If
                # no rendered metric depth is available, keep DA3's local depth
                # variation but anchor its median to a conservative scene depth.
                if using_relative_target_depth:
                    z = z / fill.clamp_min(1e-6) * float(default_depth)
        z = z * float(seed_scale)

        x_cam = (2.0 * (xs + 0.5) / float(w) - 1.0) * math.tan(float(viewpoint_cam.FoVx) * 0.5) * z
        y_cam = (1.0 - 2.0 * (ys + 0.5) / float(h)) * math.tan(float(viewpoint_cam.FoVy) * 0.5) * z
        pts_cam = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=1)
        c2w = torch.inverse(viewpoint_cam.world_view_transform.to(device))
        pts_world = pts_cam @ c2w
        pts_world = pts_world[:, :3] / pts_world[:, 3:4].clamp_min(1e-6)
        valid = torch.isfinite(pts_world).all(dim=1)
        if not valid.any():
            return None, None
        return pts_world[valid], colors[valid]


def masked_scale_shift_depth_loss(render_depth, target_depth, mask, min_mask_area=0.0005, mask_dilate=3):
    """Scale/shift invariant depth loss, evaluated only inside dynamic masks.

    Depth Anything gives relative depth; rendered Gaussian depth has scene scale.
    We fit target ~= a * render + b over valid masked pixels, then use L1.
    """
    if render_depth is None or target_depth is None or mask is None:
        return None
    if render_depth.dim() == 2:
        render_depth = render_depth.unsqueeze(0)
    if target_depth.dim() == 2:
        target_depth = target_depth.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    render_depth = render_depth[:1].float()
    target_depth = target_depth[:1].float().to(render_depth.device)
    mask = mask[:1].float().to(render_depth.device)
    if render_depth.shape[-2:] != target_depth.shape[-2:]:
        target_depth = F.interpolate(target_depth[None], size=render_depth.shape[-2:], mode="bilinear", align_corners=False)[0]
    if render_depth.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask[None], size=render_depth.shape[-2:], mode="bilinear", align_corners=False)[0]
    if mask_dilate and mask_dilate > 1:
        k = int(mask_dilate)
        if k % 2 == 0:
            k += 1
        mask = F.max_pool2d(mask[None], kernel_size=k, stride=1, padding=k // 2)[0]
    valid = (mask > 0.05) & torch.isfinite(render_depth) & torch.isfinite(target_depth) & (target_depth > 0)
    if valid.float().mean() < min_mask_area or valid.sum() < 64:
        return None
    x = render_depth[valid]
    y = target_depth[valid]
    x_mean = x.mean()
    y_mean = y.mean()
    var = ((x - x_mean) ** 2).mean().clamp_min(1e-6)
    scale = ((x - x_mean) * (y - y_mean)).mean() / var
    shift = y_mean - scale * x_mean
    aligned = scale * render_depth + shift
    return (torch.abs(aligned[valid] - target_depth[valid]) * mask[valid].clamp_min(0.05)).mean()


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter,timer, start_time):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    video_cams = None

    num_traincams = 1
    if dataset.loader != 'nerfies': # for multi-view setting
        num_traincams = int(len(train_cams) / scene.maxtime)
    
        camera_centers = []
        for i in range(num_traincams):
            camera_centers.append(train_cams[i*scene.maxtime].camera_center.cpu().numpy())
        camera_centers = np.array(camera_centers)
        cam_dists = calculate_distances(camera_centers)
        sorted_dists = np.unique(cam_dists)
        min_dist = sorted_dists[int(sorted_dists.shape[0] * 0.5)]

        last_camera_index = 0
    
    cam_no_list = list(set(c.cam_no for c in train_cams))
    print("train cameras:", cam_no_list)
    if dataset.loader in ['nerfies']:  # single-view
        loss_list = np.zeros([num_traincams, scene.maxtime]) + 100  # pick frames that have not yet been sampled
    else:  # n3v, technicolor, etc.
        loss_list = np.zeros([max(cam_no_list) + 1, scene.maxtime])
        for c in cam_no_list:
            loss_list[c] = 100

    ssim_cnt = 0
    sampled_frame_no = None
    prev_num_pts = 0

    # We sort training images to sample image of the desired camera number and frame.
    if dataset.loader not in ['nerfies']:
        train_cams = sorted(train_cams, key=lambda x: (x.cam_no, x.frame_no))

    dynamic_frame_candidates = []
    if (getattr(dataset, "use_dynamic_masks", False)
            and getattr(opt, "dynamic_frame_sample_prob", 0.0) > 0
            and dataset.loader not in ['nerfies']):
        dynamic_frame_candidates, dynamic_frame_scores = find_dynamic_frame_candidates(
            train_cams,
            min_area=getattr(opt, "dynamic_frame_sample_min_area", 0.0001),
            threshold=max(0.5, getattr(opt, "dynamic_component_threshold", 0.35)),
        )
        print(f"Dynamic-frame oversampling candidates: {len(dynamic_frame_candidates)} frames")

    motion_frame_candidates = []
    if (getattr(dataset, "use_motion_priors", False)
            and getattr(opt, "motion_prior_frame_sample_prob", 0.0) > 0
            and dataset.loader not in ['nerfies']):
        motion_frame_candidates, motion_frame_scores = find_sidecar_frame_candidates(
            train_cams,
            "motion_prior_path",
            min_area=getattr(opt, "motion_prior_frame_sample_min_area", 0.00005),
            threshold=getattr(opt, "motion_prior_threshold", 0.35),
        )
        print(f"Motion-prior oversampling candidates: {len(motion_frame_candidates)} frames")

    stereo_seed_pairs = {}
    if getattr(opt, "use_stereo_mask_seed_points", False) and dataset.loader not in ['nerfies']:
        by_frame = {}
        for cam in train_cams:
            by_frame.setdefault(int(cam.frame_no), {})[int(cam.cam_no)] = cam
        for frame_no, cams in by_frame.items():
            if 0 in cams and 1 in cams:
                stereo_seed_pairs[frame_no] = (cams[0], cams[1])
        print(f"Stereo mask seed pairs: {len(stereo_seed_pairs)} frames")

    viewpoint_stack = train_cams
    method = None

    start_time = time()
    for iteration in range(first_iter, final_iter+1):             
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if getattr(opt, "max_points", 0) > 0 and gaussians.get_xyz.shape[0] > opt.max_points:
            gaussians.enforce_max_points(opt.max_points)
            prev_num_pts = 0

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # opt.batch_size = 2
        ### Instead of the complex process below, simply training on random frames will also work well. If you follow this, comment out the `train_cams` sorting process above.
        if dataset.loader == 'nerfies':
            frame_set = np.random.choice(range(math.ceil(len(viewpoint_stack) / 2)), size=max(opt.batch_size // 2, 1))
            viewpoint_cams = [viewpoint_stack[(f*2) % scene.maxtime] for f in frame_set] + \
                             [viewpoint_stack[(f*2+1) % scene.maxtime] for f in frame_set]
        else:
            # Pick camera
            method = "random" if iteration < opt.random_until or iteration % 2 == 1 else "by_error"

            cam_no = []
            for _ in range(opt.batch_size):
                last_camera_index = sample_camera(cam_dists, last_camera_index, min_dist)
                cam_no.append(last_camera_index)
            
            forced_frame_no = sampled_frame_no
            if forced_frame_no is None:
                r = random.random()
                motion_prob = float(getattr(opt, "motion_prior_frame_sample_prob", 0.0))
                dynamic_prob = float(getattr(opt, "dynamic_frame_sample_prob", 0.0))
                if len(motion_frame_candidates) > 0 and r < motion_prob:
                    forced_frame_no = np.random.choice(motion_frame_candidates, size=opt.batch_size)
                elif len(dynamic_frame_candidates) > 0 and r < motion_prob + dynamic_prob:
                    forced_frame_no = np.random.choice(dynamic_frame_candidates, size=opt.batch_size)
            viewpoint_cams, sampled_cam_no, sampled_frame_no = image_sampler(method=method, loader=viewpoint_stack, loss_list=loss_list, batch_size=opt.batch_size, \
                cam_no=cam_no, frame_no=forced_frame_no, total_num_frames=scene.maxtime)
            if iteration >= opt.random_until and opt.num_multiview_ssim > 0 and iteration % 50 < opt.num_multiview_ssim:
                sampled_frame_no = sampled_frame_no  # reuse sampled frame (num_multiview_ssim) times
            else:
                sampled_frame_no = None
        ###
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        dynamic_masks = []
        motion_priors = []
        target_depths = []
        render_depths = []
        stereo_losses = []
        layered_rgb_losses = []
        layered_class_losses = []
        mask_densify_boost_list = []
        motion_densify_boost_list = []
        seed_xyz_list = []
        seed_rgb_list = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        cam_no_list, frame_no_list = [], []
        for viewpoint_cam in viewpoint_cams:
            if type(viewpoint_cam.original_image) == type(None):
                viewpoint_cam.load_image()  # for lazy loading (to avoid OOM issue)
            cam_no = viewpoint_cam.cam_no
            frame_no = viewpoint_cam.frame_no
            cam_no_list.append(cam_no)
            frame_no_list.append(frame_no)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, cam_no=cam_no, iter=iteration, \
                num_down_emb_c=hyper.min_embeddings, num_down_emb_f=hyper.min_embeddings)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            render_depth = render_pkg.get("depth", None)
            render_depths.append(render_depth)

            if opt.lambda_stereo_consistency > 0 and render_depth is not None:
                right_cam = make_synthetic_right_camera(viewpoint_cam, opt.stereo_baseline)
                right_pkg = render(right_cam, gaussians, pipe, background, cam_no=cam_no, iter=iteration, \
                    num_down_emb_c=hyper.min_embeddings, num_down_emb_f=hyper.min_embeddings)
                stereo_loss = stereo_consistency_loss(
                    image,
                    right_pkg["render"],
                    render_depth,
                    viewpoint_cam,
                    right_cam,
                    occlusion_tolerance=opt.stereo_occlusion_tolerance,
                )
                if stereo_loss is not None:
                    stereo_losses.append(stereo_loss)

            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            if getattr(dataset, "use_dynamic_masks", False) and getattr(viewpoint_cam, "dynamic_mask", None) is not None:
                dynamic_mask = viewpoint_cam.dynamic_mask.cuda().float().clamp(0.0, 1.0)
            else:
                dynamic_mask = torch.zeros_like(gt_image[:1])
            dynamic_masks.append(dynamic_mask.unsqueeze(0))
            if getattr(dataset, "use_motion_priors", False) and getattr(viewpoint_cam, "motion_prior", None) is not None:
                motion_prior = viewpoint_cam.motion_prior.cuda().float().clamp(0.0, 1.0)
            else:
                motion_prior = torch.zeros_like(gt_image[:1])
            motion_priors.append(motion_prior.unsqueeze(0))
            if getattr(dataset, "use_depth_maps", False) and getattr(viewpoint_cam, "depth_map", None) is not None:
                target_depth = viewpoint_cam.depth_map.cuda().float()
            else:
                target_depth = None
            target_depths.append(target_depth)

            if getattr(opt, "use_mask_guided_densification", False) and getattr(dataset, "use_dynamic_masks", False):
                boost_mask = project_gaussians_into_mask(
                    render_pkg, viewpoint_cam, dynamic_mask,
                    threshold=getattr(opt, "mask_densify_threshold", 0.25),
                )
                if boost_mask is not None:
                    mask_densify_boost_list.append(boost_mask)

            if getattr(opt, "use_motion_prior_densification", False) and getattr(dataset, "use_motion_priors", False):
                boost_mask = project_gaussians_into_mask(
                    render_pkg, viewpoint_cam, motion_prior,
                    threshold=getattr(opt, "motion_prior_threshold", 0.35),
                )
                if boost_mask is not None:
                    motion_densify_boost_list.append(boost_mask)

            if (getattr(opt, "use_mask_seed_points", False)
                    and getattr(dataset, "use_dynamic_masks", False)
                    and iteration <= getattr(opt, "mask_seed_until_iter", 0)
                    and iteration % max(1, getattr(opt, "mask_seed_interval", 500)) == 0):
                seed_xyz, seed_rgb = sample_mask_seed_points(
                    viewpoint_cam, gt_image, dynamic_mask, render_depth, target_depth=target_depth,
                    points_per_frame=getattr(opt, "mask_seed_points_per_frame", 32),
                    threshold=getattr(opt, "mask_seed_threshold", 0.35),
                    seed_scale=getattr(opt, "mask_seed_depth_scale", 0.95),
                    default_depth=getattr(opt, "mask_seed_default_depth", 2.0),
                )
                if seed_xyz is not None:
                    seed_xyz_list.append(seed_xyz)
                    seed_rgb_list.append(seed_rgb)

            if opt.use_layered_fg_bg and iteration >= opt.layered_min_iter and getattr(dataset, "use_dynamic_masks", False):
                if opt.lambda_layered_rgb > 0 and iteration % max(1, opt.layered_rgb_interval) == 0:
                    black_background = torch.zeros_like(background)
                    bg_pkg = render(viewpoint_cam, gaussians, pipe, background, cam_no=cam_no, iter=iteration, \
                        num_down_emb_c=hyper.min_embeddings, num_down_emb_f=hyper.min_embeddings, layer_mode="background")
                    fg_pkg = render(viewpoint_cam, gaussians, pipe, black_background, cam_no=cam_no, iter=iteration, \
                        num_down_emb_c=hyper.min_embeddings, num_down_emb_f=hyper.min_embeddings, layer_mode="foreground")
                    layered_rgb_losses.append(layered_rgb_separation_loss(
                        bg_pkg["render"], fg_pkg["render"], gt_image, dynamic_mask, background
                    ))
                if opt.lambda_layered_depth_class > 0 and iteration % max(1, opt.layered_class_interval) == 0:
                    class_loss = project_gaussian_layer_depth_class_loss(
                        render_pkg, viewpoint_cam, gaussians, dynamic_mask, target_depth, render_depth,
                        close_thresh=opt.layered_depth_close_thresh,
                        use_depth_gate=opt.layered_use_depth_gate,
                    )
                    if class_loss is not None:
                        layered_class_losses.append(class_loss)

            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        if (getattr(opt, "use_stereo_mask_seed_points", False)
                and getattr(dataset, "use_dynamic_masks", False)
                and iteration <= getattr(opt, "mask_seed_until_iter", 0)
                and iteration % max(1, getattr(opt, "mask_seed_interval", 500)) == 0):
            for fno in sorted(set(int(f) for f in frame_no_list)):
                pair = stereo_seed_pairs.get(fno)
                if pair is None:
                    continue
                seed_xyz, seed_rgb = sample_stereo_mask_seed_points(
                    pair[0], pair[1],
                    points_per_component=getattr(opt, "mask_seed_points_per_component", 8),
                    threshold=getattr(opt, "mask_seed_threshold", 0.35),
                    min_area=getattr(opt, "mask_seed_min_component_area", 4),
                    max_area=getattr(opt, "mask_seed_max_component_area", 4000),
                    y_tolerance=getattr(opt, "mask_seed_y_tolerance", 12.0),
                )
                if seed_xyz is not None:
                    seed_xyz_list.append(seed_xyz)
                    seed_rgb_list.append(seed_rgb)
        
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        dynamic_mask_tensor = torch.cat(dynamic_masks,0)
        motion_prior_tensor = torch.cat(motion_priors,0)

        abs_err_tensor = torch.abs(image_tensor - gt_image_tensor).mean(dim=1, keepdim=True)
        if getattr(dataset, "use_dynamic_masks", False) and opt.dynamic_loss_weight > 0:
            abs_err = abs_err_tensor
            dyn = dynamic_mask_tensor
            if opt.dynamic_loss_balance:
                # Tiny fast objects can occupy very little area; normalize the mask so they still affect optimization.
                dyn_area = dyn.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
                dyn_weight = (dyn / dyn_area).clamp(max=opt.dynamic_loss_max_weight)
            else:
                dyn_weight = dyn
            pixel_weight = 1.0 + opt.dynamic_loss_weight * dyn_weight
            Ll1_map = abs_err * pixel_weight
            Ll1_items = Ll1_map.mean(dim=(1, 2, 3)).detach()
            Ll1 = Ll1_map.mean()
        else:
            Ll1 = l1_loss(image_tensor, gt_image_tensor, keepdim=True)
            Ll1_items = Ll1.detach()
            Ll1 = Ll1.mean()
        if opt.lambda_dssim > 0. and type(sampled_frame_no) != type(None) or (method == "by_error" and (iteration % 10 == 0) and opt.num_multiview_ssim==0):
            ssim_value, ssim_map = ssim(image_tensor, gt_image_tensor)
            Lssim = (1 - ssim_value) / 2
            loss = Ll1 + opt.lambda_dssim * Lssim
        else:
            loss = Ll1

        if (getattr(dataset, "use_dynamic_masks", False)
                and getattr(opt, "dynamic_component_loss_weight", 0.0) > 0):
            component_loss = component_normalized_dynamic_loss(
                abs_err_tensor,
                dynamic_mask_tensor,
                threshold=getattr(opt, "dynamic_component_threshold", 0.35),
                min_area=getattr(opt, "dynamic_component_min_area", 4),
                max_area=getattr(opt, "dynamic_component_max_area", 6000),
                max_components=getattr(opt, "dynamic_component_max_components", 16),
            )
            if component_loss is not None:
                loss = loss + float(opt.dynamic_component_loss_weight) * component_loss

        if (getattr(dataset, "use_motion_priors", False)
                and getattr(opt, "motion_prior_loss_weight", 0.0) > 0):
            mp = motion_prior_tensor.clamp(0.0, 1.0)
            mp_bin = (mp >= float(getattr(opt, "motion_prior_threshold", 0.35))).to(mp.dtype)
            area = mp_bin.mean(dim=(1, 2, 3), keepdim=True)
            valid_area = (
                (area >= float(getattr(opt, "motion_prior_min_area", 0.00005)))
                & (area <= float(getattr(opt, "motion_prior_max_area", 0.05)))
            ).to(mp.dtype)
            weight_map = mp * valid_area
            denom = weight_map.sum(dim=(1, 2, 3)).clamp_min(1.0)
            per_item = (abs_err_tensor * weight_map).sum(dim=(1, 2, 3)) / denom.view(-1)
            if valid_area.sum() > 0:
                loss = loss + float(opt.motion_prior_loss_weight) * per_item[valid_area.view(-1) > 0].mean()

        if getattr(dataset, "use_depth_maps", False) and opt.lambda_depth_mask > 0:
            depth_losses = []
            for rd, td, dm in zip(render_depths, target_depths, dynamic_masks):
                dloss = masked_scale_shift_depth_loss(
                    rd,
                    td,
                    dm.squeeze(0),
                    min_mask_area=opt.depth_loss_min_mask_area,
                    mask_dilate=opt.depth_loss_mask_dilate,
                )
                if dloss is not None:
                    depth_losses.append(dloss)
            if len(depth_losses) > 0:
                loss = loss + opt.lambda_depth_mask * torch.stack(depth_losses).mean()

        if opt.lambda_stereo_consistency > 0 and len(stereo_losses) > 0:
            loss = loss + opt.lambda_stereo_consistency * torch.stack(stereo_losses).mean()

        if opt.use_layered_fg_bg and iteration >= opt.layered_min_iter:
            if opt.lambda_layered_rgb > 0 and len(layered_rgb_losses) > 0:
                loss = loss + opt.lambda_layered_rgb * torch.stack(layered_rgb_losses).mean()
            if opt.lambda_layered_depth_class > 0 and len(layered_class_losses) > 0:
                loss = loss + opt.lambda_layered_depth_class * torch.stack(layered_class_losses).mean()
            if opt.lambda_fg_scale > 0:
                scale_loss = foreground_scale_regularizer(gaussians, max_scale=opt.fg_scale_max)
                if scale_loss is not None:
                    loss = loss + opt.lambda_fg_scale * scale_loss

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        for i in range(len(Ll1_items)):
            loss_list[cam_no_list[i], frame_no_list[i]] = Ll1_items[i].item()

        # use l1 instead of opacity reset
        if opt.opacity_l1_coef_fine > 0.:
            loss += opt.opacity_l1_coef_fine * torch.sigmoid(gaussians._opacity.mean())

        # embedding reg using knn (https://github.com/JonathonLuiten/Dynamic3DGaussians)
        if prev_num_pts != gaussians._xyz.shape[0]:
            neighbor_sq_dist, neighbor_indices = o3d_knn(gaussians._xyz.detach().cpu().numpy(), 20)
            neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
            neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
            neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
            prev_num_pts = gaussians._xyz.shape[0]
        
        emb = gaussians._embedding[:,None,:].repeat(1,20,1)
        emb_knn = gaussians._embedding[neighbor_indices]
        loss += opt.reg_coef * weighted_l2_loss_v2(emb, emb_knn, neighbor_weight)

        # smoothness reg on temporal embeddings
        if opt.coef_tv_temporal_embedding > 0:
            weights = gaussians._deformation.weight
            N, C = weights.shape
            first_difference = weights[1:,:] - weights[N-1,:]
            second_difference = first_difference[1:,:] - first_difference[N-2,:]
            loss += opt.coef_tv_temporal_embedding * torch.square(second_difference).mean()

        
        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at iteration {iteration}; skipping optimizer step")
            gaussians.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        loss.backward()
        for param in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                      gaussians._scaling, gaussians._rotation, gaussians._opacity,
                      gaussians._embedding, gaussians._foreground_logits]:
            if getattr(param, "grad", None) is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(list(gaussians._deformation.get_mlp_parameters()), max_norm=10.0)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        if getattr(opt, "use_mask_guided_densification", False) and len(mask_densify_boost_list) > 0:
            with torch.no_grad():
                boost_mask = torch.stack(mask_densify_boost_list, dim=0).any(dim=0)
                if boost_mask.any():
                    viewspace_point_tensor_grad[boost_mask] *= float(getattr(opt, "mask_densify_grad_boost", 4.0))
        if getattr(opt, "use_motion_prior_densification", False) and len(motion_densify_boost_list) > 0:
            with torch.no_grad():
                boost_mask = torch.stack(motion_densify_boost_list, dim=0).any(dim=0)
                if boost_mask.any():
                    viewspace_point_tensor_grad[boost_mask] *= float(getattr(opt, "motion_prior_densify_grad_boost", 2.0))
        iter_end.record()

        if iteration in saving_iterations:
            elapsed_time = time()
            
            total_time_seconds = elapsed_time - start_time
            hours, remainder = divmod(total_time_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            with open(os.path.join(args.model_path, 'training_time.txt'), 'a') as file:
                file.write(f'Iteration {iteration}: {total_time_seconds} seconds ... {int(hours)}h {int(minutes)}m {seconds}sec  points: {gaussians._xyz.shape[0]}\n')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):

                    render_training_image(scene, gaussians, test_cams, render, pipe, background, iteration-1,timer.get_elapsed_time())

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
  
                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                # Prune before densifying on iterations where both fire, so the
                # current batch's dynamic protect_mask still has the same length
                # as the Gaussian tensor it protects.
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    protect_mask = None
                    if getattr(opt, "protect_dynamic_pruning", False) and len(mask_densify_boost_list) > 0:
                        protect_mask = torch.stack(mask_densify_boost_list, dim=0).any(dim=0)
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, protect_mask=protect_mask)
                    if getattr(opt, "max_points", 0) > 0 and gaussians.get_xyz.shape[0] > opt.max_points:
                        gaussians.enforce_max_points(opt.max_points)
                        prev_num_pts = 0

                    if opt.reset_opacity_ratio > 0 and iteration % opt.pruning_interval == 0:
                        gaussians.reset_opacity(opt.reset_opacity_ratio)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, opt.max_points)
                    if getattr(opt, "max_points", 0) > 0 and gaussians.get_xyz.shape[0] > opt.max_points:
                        gaussians.enforce_max_points(opt.max_points)
                        prev_num_pts = 0

            # Seed insertion must happen after all per-render visibility/radii
            # bookkeeping for this iteration. Adding Gaussians before
            # max_radii2D/add_densification_stats makes those masks shorter than
            # the parameter tensors and caused shape mismatches at seed iters.
            if (getattr(opt, "use_mask_seed_points", False)
                    and len(seed_xyz_list) > 0
                    and (getattr(opt, "max_points", 0) <= 0 or getattr(opt, "max_points", 0) > gaussians.get_xyz.shape[0])):
                seed_xyz = torch.cat(seed_xyz_list, dim=0)
                seed_rgb = torch.cat(seed_rgb_list, dim=0)
                remaining = int(opt.max_points - gaussians.get_xyz.shape[0]) if getattr(opt, "max_points", 0) > 0 else seed_xyz.shape[0]
                if remaining > 0 and seed_xyz.shape[0] > remaining:
                    keep = torch.randperm(seed_xyz.shape[0], device=seed_xyz.device)[:remaining]
                    seed_xyz = seed_xyz[keep]
                    seed_rgb = seed_rgb[keep]
                if seed_xyz.shape[0] > 0:
                    gaussians.add_seed_points(
                        seed_xyz, seed_rgb,
                        scale=getattr(opt, "mask_seed_scale", 0.01),
                        opacity=getattr(opt, "mask_seed_opacity", 0.2),
                        foreground_prob=0.9,
                    )
                    prev_num_pts = 0

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                with torch.no_grad():
                    for param in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                                  gaussians._scaling, gaussians._rotation, gaussians._opacity,
                                  gaussians._embedding, gaussians._foreground_logits]:
                        param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    gaussians._scaling.data.clamp_(-10.0, 1.0)
                    gaussians._opacity.data.clamp_(-20.0, 20.0)
                    gaussians._foreground_logits.data.clamp_(-20.0, 20.0)
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration % 50 == 0:
                    torch.cuda.empty_cache()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, shuffle=dataset.shuffle, loader=dataset.loader, duration=hyper.total_num_frames, opt=opt)
    if getattr(dataset, "use_dynamic_masks", False):
        print(f"Using dynamic masks from '{dataset.dynamic_mask_dir}' with dynamic_loss_weight={opt.dynamic_loss_weight}")
    if getattr(dataset, "use_motion_priors", False):
        print(f"Using motion priors from '{dataset.motion_prior_dir}' with loss_weight={opt.motion_prior_loss_weight}, oversample={opt.motion_prior_frame_sample_prob}, densify={opt.use_motion_prior_densification}")
    if getattr(opt, "dynamic_component_loss_weight", 0.0) > 0:
        print(f"Using component-normalized dynamic loss: weight={opt.dynamic_component_loss_weight}, threshold={opt.dynamic_component_threshold}")
    if getattr(opt, "use_mask_guided_densification", False):
        print(f"Using mask-guided densification with boost={opt.mask_densify_grad_boost}, threshold={opt.mask_densify_threshold}")
    if getattr(opt, "protect_dynamic_pruning", False):
        print(f"Protecting dynamic-mask Gaussians from opacity pruning")
    if getattr(opt, "use_mask_seed_points", False):
        print(f"Using mask seed points: interval={opt.mask_seed_interval}, until={opt.mask_seed_until_iter}, per_frame={opt.mask_seed_points_per_frame}")
    if getattr(opt, "use_stereo_mask_seed_points", False):
        print(f"Using stereo mask seed points: per_component={opt.mask_seed_points_per_component}, y_tol={opt.mask_seed_y_tolerance}")
    if getattr(dataset, "use_depth_maps", False):
        print(f"Using depth maps from '{dataset.depth_dir}' with lambda_depth_mask={opt.lambda_depth_mask}")
    if opt.lambda_stereo_consistency > 0:
        print(f"Using synthetic stereo consistency with lambda={opt.lambda_stereo_consistency}, baseline={opt.stereo_baseline}")
    if getattr(opt, "use_layered_fg_bg", False):
        print(f"Using layered foreground/background training: rgb={opt.lambda_layered_rgb} every {opt.layered_rgb_interval} iters, depth_class={opt.lambda_layered_depth_class}, fg_scale={opt.lambda_fg_scale}")
    timer.start()
    
    start_time = time()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, opt.iterations, timer, start_time)
    end_time = time()
    
    total_time_seconds = end_time - start_time
    hours, remainder = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"training time: {int(hours)}h {int(minutes)}m {seconds}sec")


def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


        
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
     
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 5000, 7000, 14000, 20000, 30000, 45000, 60000, 80000, 100000, 120000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
