#
# Stereo rendering for E-D3DGS.
# Generates right-eye views from left cameras via horizontal baseline shift.
#

import json
import os
import re
import shutil
import subprocess
import time
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_stereo_camera(R, T, ipd=0.12, convergence_distance=0.0):
    """Create right-camera extrinsics from left-camera extrinsics."""
    R = np.asarray(R).copy()
    T = np.asarray(T).copy()

    cam_center = -R @ T
    cam_right_axis = R[:, 0]
    cam_right_axis = cam_right_axis / np.linalg.norm(cam_right_axis)
    right_center = cam_center + ipd * cam_right_axis

    if convergence_distance <= 0:
        R_right = R.copy()
    else:
        left_forward = R[:, 2]
        left_forward = left_forward / np.linalg.norm(left_forward)
        fixation_point = cam_center + convergence_distance * left_forward

        right_forward = fixation_point - right_center
        right_forward_norm = np.linalg.norm(right_forward)
        if right_forward_norm < 1e-8:
            R_right = R.copy()
        else:
            right_forward = right_forward / right_forward_norm
            y_ref = R[:, 1]
            y_ref = y_ref / np.linalg.norm(y_ref)

            right_x = np.cross(y_ref, right_forward)
            right_x_norm = np.linalg.norm(right_x)
            if right_x_norm < 1e-8:
                right_x = R[:, 0]
            else:
                right_x = right_x / right_x_norm

            right_y = np.cross(right_forward, right_x)
            right_y = right_y / np.linalg.norm(right_y)
            right_x = np.cross(right_y, right_forward)
            right_x = right_x / np.linalg.norm(right_x)
            R_right = np.stack([right_x, right_y, right_forward], axis=1)

    T_right = -(R_right.T @ right_center)
    return R_right.astype(R.dtype, copy=False), T_right.astype(T.dtype, copy=False)


def _prepare_dynamic_right_mask(view, rendering_left, dilate=3, blur=1.0, shift_px=0, threshold=0.5, min_area=0.0001, max_area=0.12):
    """Return a soft 1xHxW mask for compositing dynamic regions into the right view.

    In monocular-to-stereo rendering, dynamic masked objects are only supervised in
    the left/source view. Novel right-eye disocclusions around those objects can
    therefore look broken. This mask lets us keep dynamic objects from the left
    render while using the shifted right render elsewhere.
    """
    mask = getattr(view, "dynamic_mask", None)
    if mask is None:
        return None
    mask = mask.to(device=rendering_left.device, dtype=rendering_left.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask[:1].unsqueeze(0)
    if mask.shape[-2:] != rendering_left.shape[-2:]:
        mask = F.interpolate(mask, size=rendering_left.shape[-2:], mode="bilinear", align_corners=False)

    # Diff heatmaps or over-dilated masks can cover huge parts of the frame. That
    # causes left/right ghosting in the right eye, so gate the mask aggressively.
    mask = (mask >= threshold).to(mask.dtype)
    area = float(mask.mean().detach().cpu())
    if area < min_area or area > max_area:
        return None

    if dilate and dilate > 1:
        k = int(dilate)
        if k % 2 == 0:
            k += 1
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    if shift_px:
        mask = torch.roll(mask, shifts=int(shift_px), dims=-1)
        if shift_px > 0:
            mask[..., :shift_px] = 0
        else:
            mask[..., shift_px:] = 0
    if blur and blur > 0:
        # Cheap separable-ish blur via average pooling; enough to avoid hard seams.
        k = max(3, int(round(blur * 2)) * 2 + 1)
        mask = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)

    # Re-check after dilation; skip if it would overwrite a large chunk of the right view.
    area_after = float((mask > 0.05).to(mask.dtype).mean().detach().cpu())
    if area_after > max_area:
        return None
    return mask.clamp(0.0, 1.0).squeeze(0)


def _apply_extrinsic(view, R, T):
    view.R = R
    view.T = T
    view.world_view_transform = torch.tensor(
        getWorld2View2(R, T, view.trans, view.scale),
        dtype=view.world_view_transform.dtype,
        device=view.world_view_transform.device,
    ).transpose(0, 1)
    view.full_proj_transform = (
        view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]


def _extract_frame_index(path_or_name):
    if path_or_name is None:
        return None
    stem = os.path.splitext(os.path.basename(str(path_or_name)))[0]
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def _infer_right_frame_offset(views, right_gt_by_frame_idx):
    if not right_gt_by_frame_idx or len(views) == 0:
        return 0

    right_indices = set(right_gt_by_frame_idx.keys())
    left_indices = []
    sample_count = min(len(views), 200)
    sample_ids = np.linspace(0, len(views) - 1, num=sample_count, dtype=int)
    for sample_id in sample_ids:
        view = views[sample_id]
        left_frame_idx = _extract_frame_index(getattr(view, "image_name", None))
        if left_frame_idx is None:
            left_frame_idx = _extract_frame_index(getattr(view, "image_path", None))
        if left_frame_idx is not None:
            left_indices.append(left_frame_idx)

    if not left_indices:
        return 0

    best_offset = 0
    best_matches = -1
    for offset in range(-3, 4):
        matches = sum((idx + offset) in right_indices for idx in left_indices)
        if matches > best_matches:
            best_matches = matches
            best_offset = offset
    return best_offset


def _collect_numeric_png_indices(folder):
    indices = []
    for name in os.listdir(folder):
        if not name.endswith(".png"):
            continue
        stem = os.path.splitext(name)[0]
        m = re.search(r"(\d+)$", stem)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def _encode_stereo_videos_with_ffmpeg(left_path, right_path, stereo_path, fps=30.0, make_stereo=True):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode stereo videos, but it was not found in PATH.")

    left_ids = set(_collect_numeric_png_indices(left_path))
    right_ids = set(_collect_numeric_png_indices(right_path))
    common_ids = sorted(left_ids & right_ids)
    if not common_ids:
        raise RuntimeError(f"No common left/right PNG frame indices in {left_path} and {right_path}.")

    expected = list(range(common_ids[0], common_ids[-1] + 1))
    if common_ids != expected:
        raise RuntimeError("Left/right frame indices are not contiguous; cannot encode with pattern input.")

    start = common_ids[0]
    count = len(common_ids)
    left_pattern = os.path.join(left_path, "%05d.png")
    right_pattern = os.path.join(right_path, "%05d.png")
    left_video_path = os.path.join(left_path, "left_video.mp4")
    right_video_path = os.path.join(right_path, "right_video.mp4")
    stereo_video_path = os.path.join(stereo_path, "stereo_video.mp4")

    base = ["ffmpeg", "-y", "-loglevel", "error"]
    common = [
        "-framerate",
        str(fps),
        "-start_number",
        str(start),
        "-frames:v",
        str(count),
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]

    subprocess.run(base + ["-i", left_pattern] + common + [left_video_path], check=True)
    subprocess.run(base + ["-i", right_pattern] + common + [right_video_path], check=True)

    if make_stereo:
        subprocess.run(
            base
            + ["-i", left_pattern, "-i", right_pattern, "-frames:v", str(count)]
            + ["-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]"]
            + ["-vsync", "cfr", "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", stereo_video_path],
            check=True,
        )


def _encode_comparison_grid_with_ffmpeg(render_left_path, render_right_path, gt_left_path, gt_right_path, fps=30.0):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode comparison videos, but it was not found in PATH.")

    left_ids = set(_collect_numeric_png_indices(render_left_path))
    right_ids = set(_collect_numeric_png_indices(render_right_path))
    gt_left_ids = set(_collect_numeric_png_indices(gt_left_path))
    gt_right_ids = set(_collect_numeric_png_indices(gt_right_path))
    common_ids = sorted(left_ids & right_ids & gt_left_ids & gt_right_ids)
    if not common_ids:
        raise RuntimeError("No common frame indices across rendered left/right and GT left/right folders.")

    expected = list(range(common_ids[0], common_ids[-1] + 1))
    if common_ids != expected:
        raise RuntimeError("Comparison frame indices are not contiguous; cannot encode with pattern input.")

    start = common_ids[0]
    count = len(common_ids)
    left_pattern = os.path.join(render_left_path, "%05d.png")
    right_pattern = os.path.join(render_right_path, "%05d.png")
    gt_left_pattern = os.path.join(gt_left_path, "%05d.png")
    gt_right_pattern = os.path.join(gt_right_path, "%05d.png")
    comparison_video_path = os.path.join(os.path.dirname(render_left_path), "comparison_stereo_vs_gt.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        str(start),
        "-i",
        left_pattern,
        "-framerate",
        str(fps),
        "-start_number",
        str(start),
        "-i",
        right_pattern,
        "-framerate",
        str(fps),
        "-start_number",
        str(start),
        "-i",
        gt_left_pattern,
        "-framerate",
        str(fps),
        "-start_number",
        str(start),
        "-i",
        gt_right_pattern,
        "-frames:v",
        str(count),
        "-filter_complex",
        "[2:v][0:v]scale2ref=flags=lanczos[gtl][lref];"
        "[3:v][1:v]scale2ref=flags=lanczos[gtr][rref];"
        "[lref][rref]hstack=inputs=2[top];"
        "[gtl][gtr]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[v]",
        "-map",
        "[v]",
        "-vsync",
        "cfr",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        comparison_video_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to encode comparison grid video with ffmpeg: {e}") from e
    return comparison_video_path


def create_anaglyph(left, right):
    anaglyph = torch.zeros_like(left)
    anaglyph[0, :, :] = left[0, :, :]
    anaglyph[1, :, :] = right[1, :, :]
    anaglyph[2, :, :] = right[2, :, :]
    return anaglyph


def _resize_rgb01(arr, h, w):
    if arr.shape[:2] == (h, w):
        return arr
    return np.array(Image.fromarray((np.clip(arr, 0, 1) * 255.0).astype(np.uint8)).resize((w, h), Image.BILINEAR)).astype(
        np.float32
    ) / 255.0


def _psnr(pred, gt):
    mse = np.mean((np.clip(pred, 0, 1) - np.clip(gt, 0, 1)) ** 2)
    return float("inf") if mse <= 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))


def render_stereo_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    hyperparam,
    ipd=0.12,
    convergence_distance=0.0,
    output_format="side_by_side",
    source_path=None,
    fps=30.0,
    dynamic_right_composite=False,
    dynamic_right_mask_dilate=3,
    dynamic_right_mask_blur=1.0,
    dynamic_right_mask_shift_px=0,
    dynamic_right_mask_threshold=0.5,
    dynamic_right_mask_max_area=0.12,
):
    stereo_root = os.path.join(model_path, name, f"stereo_{iteration}")
    if os.path.isdir(stereo_root):
        shutil.rmtree(stereo_root)

    render_path = os.path.join(stereo_root, "renders")
    gts_path = os.path.join(stereo_root, "gt")
    gt_right_path = os.path.join(stereo_root, "gt_right")
    left_path = os.path.join(render_path, "left")
    right_path = os.path.join(render_path, "right")
    stereo_path = os.path.join(render_path, "stereo")

    os.makedirs(left_path, exist_ok=True)
    os.makedirs(right_path, exist_ok=True)
    os.makedirs(stereo_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    source_right_dir = os.path.join(source_path, "images", "right") if source_path else None
    right_gt_files = []
    if source_right_dir and os.path.isdir(source_right_dir):
        right_gt_files.extend(glob(os.path.join(source_right_dir, "*.png")))
        right_gt_files.extend(glob(os.path.join(source_right_dir, "*.jpg")))
        right_gt_files.extend(glob(os.path.join(source_right_dir, "*.jpeg")))
    right_gt_files = sorted(right_gt_files)

    right_gt_by_frame_idx = {}
    for p in right_gt_files:
        frame_idx = _extract_frame_index(p)
        if frame_idx is not None and frame_idx not in right_gt_by_frame_idx:
            right_gt_by_frame_idx[frame_idx] = p

    has_gt_right = len(right_gt_by_frame_idx) > 0
    right_frame_offset = _infer_right_frame_offset(views, right_gt_by_frame_idx) if has_gt_right else 0
    if has_gt_right:
        os.makedirs(gt_right_path, exist_ok=True)
        print(f"GT right frames: {source_right_dir} ({len(right_gt_files)} files), offset {right_frame_offset:+d}")

    t_list = []
    psnr_left_list = []
    psnr_right_list = []
    missing_left_frame_index = 0
    missing_right_gt_match = 0
    matched_right_gt = 0

    for idx, view in enumerate(tqdm(views, desc=f"Stereo rendering ({name})")):
        if type(view.original_image) == type(None):
            if name == "video":
                view.set_image()
            else:
                view.load_image()

        render_kwargs = {
            "iter": iteration,
            "num_down_emb_c": hyperparam.min_embeddings,
            "num_down_emb_f": hyperparam.min_embeddings,
        }
        cam_no = getattr(view, "cam_no", None)
        if cam_no is not None:
            render_kwargs["cam_no"] = cam_no

        _cuda_sync()
        t0 = time.time()

        rendering_left = render(view, gaussians, pipeline, background, **render_kwargs)["render"]

        R_orig = view.R.copy()
        T_orig = view.T.copy()
        wvt_orig = view.world_view_transform.clone()
        fpt_orig = view.full_proj_transform.clone()
        cc_orig = view.camera_center.clone()

        R_right, T_right = create_stereo_camera(view.R, view.T, ipd=ipd, convergence_distance=convergence_distance)
        _apply_extrinsic(view, R_right, T_right)
        rendering_right = render(view, gaussians, pipeline, background, **render_kwargs)["render"]

        view.R = R_orig
        view.T = T_orig
        view.world_view_transform = wvt_orig
        view.full_proj_transform = fpt_orig
        view.camera_center = cc_orig

        if dynamic_right_composite:
            dynamic_mask = _prepare_dynamic_right_mask(
                view,
                rendering_left,
                dilate=dynamic_right_mask_dilate,
                blur=dynamic_right_mask_blur,
                shift_px=dynamic_right_mask_shift_px,
                threshold=dynamic_right_mask_threshold,
                max_area=dynamic_right_mask_max_area,
            )
            if dynamic_mask is not None:
                rendering_right = rendering_right * (1.0 - dynamic_mask) + rendering_left * dynamic_mask

        _cuda_sync()
        t1 = time.time()
        t_list.append(t1 - t0)

        out_name = f"{idx:05d}.png"
        torchvision.utils.save_image(rendering_left, os.path.join(left_path, out_name))
        torchvision.utils.save_image(rendering_right, os.path.join(right_path, out_name))

        gt_left = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt_left, os.path.join(gts_path, out_name))

        pred_l_np = rendering_left.permute(1, 2, 0).detach().cpu().numpy()
        pred_r_np = rendering_right.permute(1, 2, 0).detach().cpu().numpy()
        gt_l_np = gt_left.permute(1, 2, 0).detach().cpu().numpy()
        h, w = pred_l_np.shape[:2]
        gt_l_np = _resize_rgb01(gt_l_np, h, w)
        psnr_left_list.append(_psnr(pred_l_np, gt_l_np))

        left_frame_idx = _extract_frame_index(getattr(view, "image_name", None))
        if left_frame_idx is None:
            left_frame_idx = _extract_frame_index(getattr(view, "image_path", None))
        right_gt_src = None
        if has_gt_right and left_frame_idx is not None:
            right_gt_src = right_gt_by_frame_idx.get(left_frame_idx + right_frame_offset)

        if right_gt_src is not None:
            gt_r_np = np.array(Image.open(right_gt_src).convert("RGB")).astype(np.float32) / 255.0
            torchvision.utils.save_image(
                torch.from_numpy(gt_r_np).permute(2, 0, 1), os.path.join(gt_right_path, out_name)
            )
            gt_r_np = _resize_rgb01(gt_r_np, h, w)
            psnr_right_list.append(_psnr(pred_r_np, gt_r_np))
            matched_right_gt += 1
        elif has_gt_right:
            if left_frame_idx is None:
                missing_left_frame_index += 1
            else:
                missing_right_gt_match += 1

        if output_format == "side_by_side":
            stereo_image = torch.cat([rendering_left, rendering_right], dim=2)
            torchvision.utils.save_image(stereo_image, os.path.join(stereo_path, out_name))
        elif output_format == "anaglyph":
            stereo_image = create_anaglyph(rendering_left, rendering_right)
            torchvision.utils.save_image(stereo_image, os.path.join(stereo_path, out_name))

    if len(t_list) == 0:
        return

    timings = np.array(t_list[5:]) if len(t_list) > 5 else np.array(t_list)
    fps_render = float(1.0 / timings.mean()) if timings.size > 0 else 0.0
    metrics = {"num_frames": len(t_list), "fps_pairs": fps_render, "right_frame_offset": int(right_frame_offset)}
    if psnr_left_list:
        metrics["psnr_left"] = float(np.mean(psnr_left_list))
    if psnr_right_list:
        metrics["psnr_right"] = float(np.mean(psnr_right_list))
    if has_gt_right:
        metrics["matched_right_gt_frames"] = int(matched_right_gt)
        metrics["missing_left_frame_index"] = int(missing_left_frame_index)
        metrics["missing_right_gt_match"] = int(missing_right_gt_match)

    with open(os.path.join(render_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _encode_stereo_videos_with_ffmpeg(
        left_path, right_path, stereo_path, fps=fps, make_stereo=(output_format == "side_by_side")
    )

    if has_gt_right and matched_right_gt > 0:
        try:
            comparison_video_path = _encode_comparison_grid_with_ffmpeg(
                left_path, right_path, gts_path, gt_right_path, fps=fps
            )
            print(f"Comparison 2x2 video (Rendered L|R over GT L|R): {comparison_video_path}")
        except RuntimeError as e:
            print(f"Skipped comparison grid video: {e}")


def render_stereo_sets(
    dataset: ModelParams,
    hyperparam: ModelHiddenParams,
    opt: OptimizationParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    skip_video: bool,
    ipd: float,
    convergence_distance: float,
    output_format: str,
    gt_source_path: str = None,
    fps: float = 30.0,
    dynamic_right_composite: bool = False,
    dynamic_right_mask_dilate: int = 3,
    dynamic_right_mask_blur: float = 1.0,
    dynamic_right_mask_shift_px: int = 0,
    dynamic_right_mask_threshold: float = 0.5,
    dynamic_right_mask_max_area: float = 0.12,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            duration=hyperparam.total_num_frames,
            loader=dataset.loader,
            opt=opt,
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        source_path = gt_source_path if gt_source_path else dataset.source_path

        if not skip_train:
            render_stereo_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                hyperparam,
                ipd=ipd,
                convergence_distance=convergence_distance,
                output_format=output_format,
                source_path=source_path,
                fps=fps,
                dynamic_right_composite=dynamic_right_composite,
                dynamic_right_mask_dilate=dynamic_right_mask_dilate,
                dynamic_right_mask_blur=dynamic_right_mask_blur,
                dynamic_right_mask_shift_px=dynamic_right_mask_shift_px,
                dynamic_right_mask_threshold=dynamic_right_mask_threshold,
                dynamic_right_mask_max_area=dynamic_right_mask_max_area,
            )
        if not skip_test:
            render_stereo_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                hyperparam,
                ipd=ipd,
                convergence_distance=convergence_distance,
                output_format=output_format,
                source_path=source_path,
                fps=fps,
                dynamic_right_composite=dynamic_right_composite,
                dynamic_right_mask_dilate=dynamic_right_mask_dilate,
                dynamic_right_mask_blur=dynamic_right_mask_blur,
                dynamic_right_mask_shift_px=dynamic_right_mask_shift_px,
                dynamic_right_mask_threshold=dynamic_right_mask_threshold,
                dynamic_right_mask_max_area=dynamic_right_mask_max_area,
            )
        if not skip_video:
            render_stereo_set(
                dataset.model_path,
                "video",
                scene.loaded_iter,
                scene.getVideoCameras(),
                gaussians,
                pipeline,
                background,
                hyperparam,
                ipd=ipd,
                convergence_distance=convergence_distance,
                output_format=output_format,
                source_path=source_path,
                fps=fps,
                dynamic_right_composite=dynamic_right_composite,
                dynamic_right_mask_dilate=dynamic_right_mask_dilate,
                dynamic_right_mask_blur=dynamic_right_mask_blur,
                dynamic_right_mask_shift_px=dynamic_right_mask_shift_px,
                dynamic_right_mask_threshold=dynamic_right_mask_threshold,
                dynamic_right_mask_max_area=dynamic_right_mask_max_area,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Stereo rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--ipd", default=0.12, type=float, help="Inter-pupillary distance in meters")
    parser.add_argument("--convergence_distance", default=0.0, type=float, help="Toe-in convergence plane distance")
    parser.add_argument(
        "--output_format",
        default="side_by_side",
        choices=["side_by_side", "anaglyph", "separate"],
        help="Stereo frame output mode",
    )
    parser.add_argument("--gt_source_path", default=None, type=str, help="Scene root containing images/right GT")
    parser.add_argument("--fps", default=30.0, type=float, help="Output video FPS")
    parser.add_argument(
        "--dynamic_right_composite",
        action="store_true",
        help="For monocular-to-stereo, composite left-rendered dynamic-mask regions into the generated right view to avoid right-eye disocclusion holes on moving objects.",
    )
    parser.add_argument("--dynamic_right_mask_dilate", default=3, type=int, help="Dilation kernel for dynamic right-view composite mask")
    parser.add_argument("--dynamic_right_mask_blur", default=1.0, type=float, help="Soft blur radius for dynamic right-view composite mask")
    parser.add_argument("--dynamic_right_mask_shift_px", default=0, type=int, help="Optional horizontal pixel shift applied to the composite mask; keep 0 unless you know the needed shift")
    parser.add_argument("--dynamic_right_mask_threshold", default=0.5, type=float, help="Threshold applied to dynamic mask before compositing; use high values for diff heatmaps")
    parser.add_argument("--dynamic_right_mask_max_area", default=0.12, type=float, help="Skip compositing if processed mask covers more than this fraction of the frame")

    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Stereo rendering", args.model_path)
    print(f"  IPD: {args.ipd * 100:.1f} cm")
    print(f"  Convergence distance: {args.convergence_distance} m")
    print(f"  Output format: {args.output_format}")
    if args.dynamic_right_composite:
        print(
            "  Dynamic right composite: enabled "
            f"(dilate={args.dynamic_right_mask_dilate}, blur={args.dynamic_right_mask_blur}, "
            f"shift_px={args.dynamic_right_mask_shift_px}, threshold={args.dynamic_right_mask_threshold}, "
            f"max_area={args.dynamic_right_mask_max_area})"
        )

    safe_state(args.quiet)
    render_stereo_sets(
        model.extract(args),
        hyperparam.extract(args),
        opt.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.skip_video,
        args.ipd,
        args.convergence_distance,
        args.output_format,
        gt_source_path=args.gt_source_path,
        fps=args.fps,
        dynamic_right_composite=args.dynamic_right_composite,
        dynamic_right_mask_dilate=args.dynamic_right_mask_dilate,
        dynamic_right_mask_blur=args.dynamic_right_mask_blur,
        dynamic_right_mask_shift_px=args.dynamic_right_mask_shift_px,
        dynamic_right_mask_threshold=args.dynamic_right_mask_threshold,
        dynamic_right_mask_max_area=args.dynamic_right_mask_max_area,
    )
