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


def parse_stereo_baseline_curriculum(spec, total_iters, default_baseline):
    """Parse a stereo-baseline schedule.

    Supported forms:
      - "0.3,0.6,1.0,1.5,2.0" spreads milestones across training.
      - "1:0.3,6000:0.6,12000:1.0" uses explicit iteration milestones.
    """
    spec = str(spec or "").strip()
    if not spec:
        return None

    tokens = [tok.strip() for tok in spec.replace(";", ",").split(",") if tok.strip()]
    if len(tokens) == 0:
        return None

    has_explicit_iters = any((":" in tok) or ("@" in tok) for tok in tokens)
    milestones = []
    try:
        if has_explicit_iters:
            for tok in tokens:
                sep = ":" if ":" in tok else "@"
                it_s, val_s = tok.split(sep, 1)
                milestones.append((int(float(it_s)), float(val_s)))
        else:
            values = [float(tok) for tok in tokens]
            if len(values) == 1:
                milestones = [(1, values[0])]
            else:
                total_iters = max(int(total_iters), 1)
                for i, val in enumerate(values):
                    frac = i / max(len(values) - 1, 1)
                    it = 1 + int(round(frac * (total_iters - 1)))
                    milestones.append((it, val))
    except ValueError as exc:
        raise ValueError(f"Invalid --stereo_baseline_curriculum '{spec}': {exc}") from exc

    milestones = sorted((max(1, int(it)), float(val)) for it, val in milestones)
    if len(milestones) == 0:
        return None
    if milestones[0][0] > 1:
        milestones.insert(0, (1, float(default_baseline)))
    return milestones


def stereo_baseline_at_iteration(milestones, iteration, mode="linear"):
    if not milestones:
        return None
    iteration = int(iteration)
    if iteration <= milestones[0][0]:
        return float(milestones[0][1])
    for (it0, b0), (it1, b1) in zip(milestones[:-1], milestones[1:]):
        if iteration <= it1:
            if str(mode).lower() == "step" or it1 <= it0:
                return float(b0)
            t = (iteration - it0) / float(it1 - it0)
            return float(b0 + t * (b1 - b0))
    return float(milestones[-1][1])


def project_gaussians_into_mask(render_pkg, viewpoint_cam, mask, threshold=0.25):
    """Return a bool mask over Gaussians whose projected centers land in a 2D sidecar mask."""
    means = render_pkg.get("means3D_final", None)
    radii = render_pkg.get("radii", None)
    if means is None or radii is None or mask is None:
        return None
    with torch.no_grad():
        device = means.device
        n = means.shape[0]
        h, w = int(viewpoint_cam.image_height), int(viewpoint_cam.image_width)
        mask_img = _resize_1chw(mask, (h, w)).to(device).float().clamp(0.0, 1.0)
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

    stereo_baseline_schedule = parse_stereo_baseline_curriculum(
        getattr(opt, "stereo_baseline_curriculum", ""),
        final_iter,
        getattr(opt, "stereo_baseline", 0.03),
    )
    if getattr(opt, "lambda_stereo_consistency", 0.0) > 0 and stereo_baseline_schedule:
        print(
            "Stereo baseline curriculum "
            f"({getattr(opt, 'stereo_baseline_curriculum_mode', 'linear')}): "
            + ", ".join(f"{it}:{baseline:g}" for it, baseline in stereo_baseline_schedule)
        )

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
                if len(motion_frame_candidates) > 0 and r < motion_prob:
                    forced_frame_no = np.random.choice(motion_frame_candidates, size=opt.batch_size)
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
        motion_priors = []
        stereo_losses = []
        motion_densify_boost_list = []
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

            if opt.lambda_stereo_consistency > 0 and render_depth is not None:
                stereo_baseline = stereo_baseline_at_iteration(
                    stereo_baseline_schedule,
                    iteration,
                    getattr(opt, "stereo_baseline_curriculum_mode", "linear"),
                )
                if stereo_baseline is None:
                    stereo_baseline = opt.stereo_baseline
                right_cam = make_synthetic_right_camera(viewpoint_cam, stereo_baseline)
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
            if getattr(dataset, "use_motion_priors", False) and getattr(viewpoint_cam, "motion_prior", None) is not None:
                motion_prior = viewpoint_cam.motion_prior.cuda().float().clamp(0.0, 1.0)
            else:
                motion_prior = torch.zeros_like(gt_image[:1])
            motion_priors.append(motion_prior.unsqueeze(0))

            if getattr(opt, "use_motion_prior_densification", False) and getattr(dataset, "use_motion_priors", False):
                boost_mask = project_gaussians_into_mask(
                    render_pkg, viewpoint_cam, motion_prior,
                    threshold=getattr(opt, "motion_prior_threshold", 0.35),
                )
                if boost_mask is not None:
                    motion_densify_boost_list.append(boost_mask)

            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        motion_prior_tensor = torch.cat(motion_priors,0)

        abs_err_tensor = torch.abs(image_tensor - gt_image_tensor).mean(dim=1, keepdim=True)
        Ll1 = l1_loss(image_tensor, gt_image_tensor, keepdim=True)
        Ll1_items = Ll1.detach()
        Ll1 = Ll1.mean()
        if opt.lambda_dssim > 0. and type(sampled_frame_no) != type(None) or (method == "by_error" and (iteration % 10 == 0) and opt.num_multiview_ssim==0):
            ssim_value, ssim_map = ssim(image_tensor, gt_image_tensor)
            Lssim = (1 - ssim_value) / 2
            loss = Ll1 + opt.lambda_dssim * Lssim
        else:
            loss = Ll1

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

        if opt.lambda_stereo_consistency > 0 and len(stereo_losses) > 0:
            loss = loss + opt.lambda_stereo_consistency * torch.stack(stereo_losses).mean()

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
                      gaussians._embedding]:
            if getattr(param, "grad", None) is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(list(gaussians._deformation.get_mlp_parameters()), max_norm=10.0)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
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

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
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

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                with torch.no_grad():
                    for param in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
                                  gaussians._scaling, gaussians._rotation, gaussians._opacity,
                                  gaussians._embedding]:
                        param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
                    gaussians._scaling.data.clamp_(-10.0, 1.0)
                    gaussians._opacity.data.clamp_(-20.0, 20.0)
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
    if getattr(dataset, "use_motion_priors", False):
        print(f"Using motion priors from '{dataset.motion_prior_dir}' with loss_weight={opt.motion_prior_loss_weight}, oversample={opt.motion_prior_frame_sample_prob}, densify={opt.use_motion_prior_densification}")
    if opt.lambda_stereo_consistency > 0:
        curriculum = getattr(opt, "stereo_baseline_curriculum", "")
        if curriculum:
            print(
                f"Using synthetic stereo consistency with lambda={opt.lambda_stereo_consistency}, "
                f"baseline curriculum='{curriculum}', mode={getattr(opt, 'stereo_baseline_curriculum_mode', 'linear')}"
            )
        else:
            print(f"Using synthetic stereo consistency with lambda={opt.lambda_stereo_consistency}, baseline={opt.stereo_baseline}")
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
