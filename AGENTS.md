# AGENTS.md — E-D3DGS-Stereo-FastMotion-DynamicMasks

> **Auto-generated repository map for the pi coding agent.**
> Last updated: 2026-06-06

---

## 1. Project Identity

This is a **fork of [E-D3DGS](https://github.com/JeongminB/E-D3DGS)** (ECCV 2024) — "Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting." The original paper: [arXiv 2404.03613](https://arxiv.org/abs/2404.03613).

This fork adds **three major capability layers** on top of upstream E-D3DGS:

| Layer | Description |
|-------|-------------|
| **Stereo Rendering** | Synthetic right-eye view generation via horizontal baseline shift (`render_stereo.py`), plus stereo-consistency training loss (`stereo_consistency_loss` in `train.py`) |
| **Fast-Motion / Dynamic Masks** | Per-frame dynamic-object masks for mask-weighted RGB supervision, component-normalized losses, mask-guided densification/pruning/seeding, and dynamic-frame oversampling |
| **Layered FG/BG** | Learned per-Gaussian foreground probability, layer-separated rendering, depth-guided foreground classification loss, and static-background locking |

The codebase is currently used on an **HPC cluster environment** (Imperial College RCS) with SLURM-managed GPU nodes. The primary entry point for runs is the shell script:

```
/rds/general/ephemeral/user/ka1525/ephemeral/jobs/run_ed3dgs_dynamic_masks_train_render_stereo_interactive.sh
```

This script is run interactively in a **JupyterHub GPU terminal**, not via `sbatch`.

---

## 2. Directory Structure

```
E-D3DGS-Stereo-FastMotion-DynamicMasks/
├── train.py                  # Main training script (~1224 lines)
├── render.py                 # Monocular rendering
├── render_stereo.py          # Stereo (left+right) rendering with video encoding
├── metrics.py                # Evaluation: PSNR, SSIM, LPIPS, TF, EPE, D1-all, IQ
├── convert.py                # COLMAP conversion utility
├── external.py               # Original 3DGS external utilities (SSIM, densify helpers)
├── helpers.py                # Camera setup, quaternion math, KNN, parameter I/O
├── requirements.txt          # pip dependencies
├── README.md                 # Upstream README (E-D3DGS paper info)
├── DYNAMIC_MASKS.md          # Dynamic mask usage guide
├── LICENSE.md                # License (non-commercial research only)
├── .gitignore
│
├── arguments/
│   ├── __init__.py           # Argparse groups: ModelParams, PipelineParams,
│   │                         #   OptimizationParams, ModelHiddenParams
│   ├── dynerf/               # Scene-specific configs for Neural 3D Video dataset
│   │   ├── default.py        # Base dynerf config
│   │   ├── coffee_martini.py, cook_spinach.py, cut_roasted_beef.py,
│   │   ├── flame_salmon_frag[1-4].py, flame_steak.py, sear_steak.py
│   │   └── coffee_martini_wo_cam13.py
│   ├── technicolor/          # Scene configs for Technicolor dataset
│   │   └── Birthday.py, Fabien.py, Painter.py, Theater.py, Train.py, default.py
│   └── hypernerf/            # Scene configs for HyperNeRF dataset
│       └── vrig-3dprinter.py, vrig-broom.py, vrig-chicken.py,
│           vrig-peel-banana.py, default.py
│
├── scene/
│   ├── __init__.py           # Scene class: loads datasets, manages train/test/video cameras
│   ├── cameras.py            # Camera dataclass with lazy image loading, dynamic mask,
│   │                         #   motion prior, and depth map fields
│   ├── colmap_loader.py      # COLMAP binary/text data readers
│   ├── dataset_readers.py    # Dataset loaders: Dynerf, Technicolor, Nerfies (HyperNeRF)
│   ├── deformation.py        # Deformation network (MLP): coarse+fine, temporal embeddings
│   ├── gaussian_model.py     # GaussianModel: core 3D Gaussian parameter management,
│   │                         #   training setup, densification, pruning, seed points,
│   │                         #   foreground logits, PLY I/O
│   ├── hyper_loader.py       # HyperNeRF camera/pose loading utilities
│   └── utils.py              # Scene utilities
│
├── gaussian_renderer/
│   ├── __init__.py           # Render function: deformation application, rasterization,
│   │                         #   layer rendering (fg/bg), NaN/inf clamping
│   └── network_gui.py        # Optional network GUI (not actively used)
│
├── utils/
│   ├── camera_utils.py       # Camera creation from dataset info
│   ├── extra_utils.py        # o3d_knn, image_sampler, sample_camera, frame scheduling
│   ├── general_utils.py      # safe_state, inverse_sigmoid, build_rotation, LR scheduling
│   ├── graphics_utils.py     # Projection matrices, world2view, FoV utilities
│   ├── image_utils.py        # PSNR computation
│   ├── loss_utils.py         # L1, L2, SSIM, LPIPS losses
│   ├── params_utils.py       # Config merging (merge_hparams)
│   ├── pose_utils.py         # Pose interpolation utilities
│   ├── scene_utils.py        # render_training_image (intermediate render logging)
│   ├── sh_utils.py           # Spherical harmonics utilities
│   ├── system_utils.py       # searchForMaxIteration, mkdir_p
│   ├── timer.py              # Simple Timer class
│   └── *.TTF                 # Times font files for image rendering
│
├── script/
│   ├── colmap_setup.sh           # COLMAP installation script
│   ├── pre_n3v.py                # Preprocess Neural 3D Video dataset
│   ├── pre_technicolor.py        # Preprocess Technicolor dataset
│   ├── pre_hypernerf.py          # Preprocess HyperNeRF dataset
│   ├── downsample_point.py       # Downsample COLMAP dense point clouds
│   ├── generate_dynamic_masks.py # Main dynamic mask generation (median/temporal/combined)
│   ├── generate_ball_masks_video010.sh  # Ball-mask-specific generation script
│   ├── generate_masks_video010.sh       # General mask generation for video010
│   ├── make_small_object_masks_from_diffs.py  # Convert motion diffs to compact component masks
│   ├── generate_motion_priors.py  # Generate camera-compensated residual motion priors
│   ├── generate_da3_depth_maps.py # Generate Depth Anything V3 depth maps
│   ├── generate_da3_depth_video010.sh  # DA3 generation wrapper
│   └── thirdparty/               # Preprocessing utilities
│       ├── arguments.py, colmap_loader.py, general_utils.py
│       ├── helper3dg.py, my_utils.py, pre_colmap.py
│
├── submodules/
│   ├── diff-gaussian-rasterization/  # CUDA rasterizer (built with pip install -e .)
│   │   ├── cuda_rasterizer/          # CUDA kernels (forward/backward)
│   │   └── third_party/glm/          # GLM math library
│   └── simple-knn/                   # CUDA KNN for initialization (built with pip install -e .)
│
├── lpipsPyTorch/                     # LPIPS perceptual loss (vendored)
├── output/                           # Default output directory for training results
├── train_hypernerf.sh                # Convenience training script (HyperNeRF)
├── train_n3v.sh                      # Convenience training script (N3V)
├── train_technicolor.sh              # Convenience training script (Technicolor)
└── teaser.gif                        # Project teaser animation
```

---

## 3. Architecture Overview

### 3.1 Core Pipeline

```
Data → Scene → GaussianModel + deform_network → GaussianRenderer → Loss → Backprop
```

1. **Data Loading** (`scene/__init__.py`): Scene reads dataset via `dataset_readers.py`, creates `Camera` objects for train/test/video splits. Supports `dynerf`, `technicolor`, and `nerfies` loaders.

2. **Gaussian Model** (`scene/gaussian_model.py`): Manages all differentiable parameters:
   - `_xyz` — 3D positions
   - `_features_dc`, `_features_rest` — Spherical harmonics coefficients (color)
   - `_scaling`, `_rotation` — Covariance parameters
   - `_opacity` — Per-Gaussian opacity
   - `_embedding` — Per-Gaussian latent embedding (E-D3DGS innovation)
   - `_foreground_logits` — Per-Gaussian foreground probability (fork addition)

3. **Deformation Network** (`scene/deformation.py`): MLP that predicts per-Gaussian deformation from temporal + Gaussian embeddings. Architecture:
   - Temporal embedding: learned grid `(max_embeddings × temporal_embedding_dim)` interpolated by frame time
   - Gaussian embedding: per-point latent `(gaussian_embedding_dim)`
   - Concatenated embeddings → feature MLP → position/scale/rotation/opacity/SH deltas
   - Coarse-to-fine progressive temporal resolution
   - Two deformation stages: coarse net → fine net

4. **Rendering** (`gaussian_renderer/__init__.py`): Applies deformation, then CUDA rasterization. Supports:
   - Standard rendering
   - Foreground-only rendering (`layer_mode="foreground"`)
   - Background-only rendering (`layer_mode="background"`)
   - Static background locking (blends deformed params toward original by `1 - fg_prob`)
   - NaN/inf value clamping after deformation

5. **Training Loop** (`train.py::scene_reconstruction`): Main 30K–80K iteration loop with:
   - Frame/camera sampling (random vs. by-error, with oversampling for dynamic/motion frames)
   - Loss computation (L1, SSIM, stereo consistency, dynamic mask weighted, component-normalized, depth, layered FG/BG)
   - Gradient accumulation and densification (clone/split/prune)
   - Seed point insertion from dynamic masks (monocular and stereo triangulated)
   - Checkpoint and intermediate render saving

### 3.2 Fork-Specific Innovations

All new features are in `train.py` and `gaussian_renderer/__init__.py`:

| Feature | CLI Flag(s) | Description |
|---------|-------------|-------------|
| Dynamic mask loss | `--use_dynamic_masks`, `--dynamic_loss_weight` | Weighted L1 where foreground pixels get `1 + dynamic_loss_weight * mask` multiplier |
| Component-normalized loss | `--dynamic_component_loss_weight` | Averages error per connected component so tiny objects aren't drowned out |
| Dynamic frame oversampling | `--dynamic_frame_sample_prob` | Oversamples frames with motion content during training |
| Mask-guided densification | `--use_mask_guided_densification` | Boosts densification gradients for Gaussians projecting into dynamic masks |
| Pruning protection | `--protect_dynamic_pruning` | Prevents opacity pruning of Gaussians inside dynamic masks |
| Mask seed points | `--use_mask_seed_points` | Inserts new Gaussians at dynamic mask locations (monocular depth back-projection) |
| Stereo mask seed points | `--use_stereo_mask_seed_points` | Triangulates seed points from matched left/right mask components |
| Motion priors | `--use_motion_priors`, `--motion_prior_loss_weight` | 2D motion prior (Farneback or RAFT) as training signal |
| Depth supervision | `--use_depth_maps`, `--lambda_depth_mask` | Scale/shift-invariant depth loss from Depth Anything V3 inside masks |
| Stereo consistency | `--lambda_stereo_consistency` | Forward-warps left render to right view and penalizes disagreement |
| Layered FG/BG | `--use_layered_fg_bg` | Trains separate foreground/background layers with depth-guided classification |
| Static background locking | `--layer_static_background` | Locks background Gaussians to original positions after `min_iter` |

---

## 4. Key Files by Function

### Training Entry Point
- **`train.py`** — Main script. Parses args, merges configs, runs `training()`. The `scene_reconstruction()` function (L90–L1048) contains the entire training loop.

### Rendering Entry Points
- **`render.py`** — Monocular rendering for train/test/video cameras. Saves renders, GTs, and optionally dynamic mask overlays.
- **`render_stereo.py`** — Stereo pair rendering. Generates left+right views via baseline shift, supports anaglyph/side-by-side/separate output, encodes videos with ffmpeg, optionally composites dynamic mask regions into the right view.

### Evaluation
- **`metrics.py`** — Computes PSNR, SSIM, LPIPS (VGG + Alex), temporal flickering score (VBench-style), proxy optical-flow EPE, proxy D1-all stereo metric, and optional MUSIQ IQ-Score via pyiqa. Handles both mono and stereo render outputs.

### Deformation & Model
- **`scene/deformation.py`** — `deform_network` class. Two-stage (coarse+fine) MLP deformation with temporal + Gaussian embeddings. Key parameters: `temporal_embedding_dim=256`, `gaussian_embedding_dim=32`, `net_width=64/128`, `defor_depth=1`.
- **`scene/gaussian_model.py`** — `GaussianModel` class. Manages all Gaussians, densification, pruning, seed point insertion, PLY I/O. The `_foreground_logits` parameter and `add_seed_points()` method are fork additions.

### Argument System
- **`arguments/__init__.py`** — Four `ParamGroup` classes:
  - `ModelParams`: data paths, dataset loader, dynamic/motion/depth mask dirs
  - `PipelineParams`: debug, SH/compute_cov3D flags
  - `OptimizationParams`: all hyperparameters including fork-specific ones (dynamic loss weights, stereo params, layered FG/BG params, seed params)
  - `ModelHiddenParams`: deformation network architecture, temporal embedding config, coarse-to-fine settings

### Scene Assembly
- **`scene/__init__.py`** — `Scene` class. Loads dataset, initializes Gaussians from point cloud, manages camera lists with resolution scaling. Supports `dynerf`, `technicolor`, `nerfies` loaders.
- **`scene/cameras.py`** — `Camera` nn.Module. Stores extrinsics, intrinsics, time, lazy-loaded images, dynamic masks, motion priors, and depth maps.

### Data Preprocessing Scripts
- **`script/generate_dynamic_masks.py`** — Generates per-frame dynamic masks via temporal median background subtraction, frame differencing, and connected-component filtering.
- **`script/make_small_object_masks_from_diffs.py`** — Post-processes motion difference heatmaps into compact binary component masks.
- **`script/generate_motion_priors.py`** — Generates camera-compensated residual motion maps (Farneback or RAFT optical flow).
- **`script/generate_da3_depth_maps.py`** — Runs Depth Anything V3 on scene images.

---

## 5. Configuration System

Training uses **Python config files** loaded via `mmcv.Config.fromfile()` and merged with CLI args using `merge_hparams()`. Configs are in `arguments/<dataset>/<scene>.py`.

Example config (`arguments/dynerf/default.py`):
```python
ModelParams = dict(loader="dynerf")
ModelHiddenParams = dict(
    defor_depth=1, net_width=128,
    use_coarse_temporal_embedding=True,
    c2f_temporal_iter=10000,
    deform_from_iter=5000,
    total_num_frames=300,
)
OptimizationParams = dict(
    iterations=80_000, maxtime=300,
    lambda_dssim=1, num_multiview_ssim=5,
    reg_coef=1.0,
)
```

**Important**: Legacy upstream code uses `_base_ = './default.py'` at the top of scene configs, but `mmcv.Config.fromfile()` does NOT natively support `_base_`. This may silently fail to inherit defaults.

---

## 6. HPC Environment & Job Execution

### Environment Setup
```bash
module load tools/prod miniforge/3 CUDA/11.7.0 FFmpeg/6.0-GCCcore-12.3.0
conda activate ed3dgs-stereo
```

The conda environment `ed3dgs-stereo` requires:
- Python 3.10+ (upstream uses 3.7)
- PyTorch 1.13.1+cu116 (or compatible)
- CUDA 11.7
- The two CUDA submodules built: `diff-gaussian-rasterization` and `simple-knn`

### Primary Job Script
**`/rds/general/ephemeral/user/ka1525/ephemeral/jobs/run_ed3dgs_dynamic_masks_train_render_stereo_interactive.sh`**

This is a **bash script run interactively** in a JupyterHub GPU terminal. It is NOT submitted via `sbatch`. Key characteristics:
- Processes scenes listed in `SCENES` array (default: `video013`)
- Supports environment variable overrides for all parameters
- Pipeline stages controlled by toggles: `RUN_TRAIN`, `RUN_RENDER`, `RUN_STEREO`, `RUN_METRICS`, `GENERATE_MASKS`, `GENERATE_DEPTH`, `GENERATE_MOTION_PRIORS`
- Two enhancement modes: `conservative` (default, safe for camera-motion scenes) and `tiny_object` (aggressive, for scenes with small fast-moving objects)
- CUDA_VISIBLE_DEVICES handling with GPU UUID → index mapping
- Dataset root: `/rds/general/ephemeral/user/ka1525/ephemeral/datasets/SK/indoor`
- Output prefix: `ed3dgs-dynamic-masks-indoor`

### Common Overrides
```bash
# Generate masks, clean output, train
GENERATE_MASKS=1 CLEAN_OUTPUT=1 bash jobs/run_ed3dgs_dynamic_masks_train_render_stereo_interactive.sh video010

# Render only (skip training)
RUN_TRAIN=0 RUN_RENDER=1 RUN_STEREO=1 bash ... video010

# Different mask directory
DYNAMIC_MASK_DIR=dynamic_mask_ball_diffs bash ... video010

# Enable stereo consistency during training
USE_SYNTHETIC_STEREO_CONSISTENCY=1 bash ... video013
```

### Other Job Scripts
The `jobs/` directory contains companion scripts for related models:
- `run_ed3dgs_train_render_interactive.sh` — Vanilla E-D3DGS (no dynamic masks, no stereo)
- `run_ed3dgs_fastmotion_train_stereo_render_interactive.sh` — FastMotion variant
- `run_ed3dgs_trajectory_train_stereo_render_interactive.sh` — Trajectory variant
- `run_ed3dgs_stereo_render_interactive.sh` — Stereo render only (no training)
- `run_ed3dgs_fastmotion_render_first.sh` — Render-first variant
- `run_4dgs_*`, `run_scaffoldgs_*`, `run_spin4dgs_*`, `run_trackersplat_*` — Other model scripts

---

## 7. Data Layout Expectations

```
datasets/SK/indoor/<scene>/
├── images/                    # Camera folders with PNG frames
│   ├── cam01/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── cam02/
│   │   └── ...
│   └── ...
├── ns_output/colmap/          # COLMAP output
│   └── sparse/0/              # Sparse reconstruction
├── dynamic_masks/             # (generated) Per-frame dynamic masks
│   └── cam01/
│       ├── 0000.png
│       └── ...
├── dynamic_masks_small_from_diffs_p98_v2/  # Compact small-object masks
├── motion_priors/             # (generated) Motion prior maps
├── depth_da3/                 # (generated) Depth Anything V3 depth maps
└── points3D_downsample.ply    # (optional) Downsampled COLMAP point cloud
```

Mask files are expected to be **single-channel PNGs** with white (1.0) = dynamic foreground, black (0.0) = static background. Soft masks are supported.

---

## 8. Output Structure

```
output/<prefix>-<scene>/
├── cameras.json               # Camera metadata
├── input.ply                  # Initial point cloud
├── cfg_args                   # Full merged config (serialized Namespace)
├── training_time.txt          # Per-iteration timing log
├── results.json               # Evaluation metrics
├── per_view.json              # Per-frame metrics
├── point_cloud/
│   └── iteration_<N>/
│       ├── point_cloud.ply    # Gaussian parameters
│       └── deformation.pth    # Deformation network weights
├── test/
│   └── ours_<N>/
│       ├── renders/           # Rendered test views
│       ├── gt/                # Ground truth
│       ├── dynamic_masks/     # (if masks enabled)
│       └── video_rgb.mp4
├── train/
│   └── stereo_<N>/
│       └── renders/
│           ├── left/          # Left-eye renders
│           ├── right/         # Right-eye renders
│           ├── stereo/        # Side-by-side frames
│           ├── left_video.mp4
│           ├── right_video.mp4
│           ├── stereo_video.mp4
│           ├── metrics.json
│           ├── gt/            # GT left frames
│           └── gt_right/      # GT right frames (if available)
└── train_render/              # Intermediate training renders
```

---

## 9. Common Workflows

### Training from scratch with dynamic masks
```bash
cd /rds/general/ephemeral/user/ka1525/ephemeral/models/E-D3DGS-Stereo-FastMotion-DynamicMasks
python train.py \
  -s /path/to/scene \
  --loader dynerf \
  --images images \
  --model_path output/ed3dgs-dynamic-masks-indoor-video010 \
  --expname dynerf/video010 \
  --iterations 30000 \
  --maxtime 300 \
  --total_num_frames 300 \
  --max_points 350000 \
  -r 2 \
  --use_dynamic_masks \
  --dynamic_mask_dir dynamic_masks \
  --dynamic_loss_weight 1.0 \
  --dynamic_loss_balance \
  --dynamic_loss_max_weight 5.0
```

### Stereo rendering after training
```bash
python render_stereo.py \
  --model_path output/ed3dgs-dynamic-masks-indoor-video010 \
  --ipd 0.12 \
  --output_format side_by_side \
  --gt_source_path /path/to/scene \
  --fps 30
```

### Evaluation
```bash
python metrics.py --model_paths output/ed3dgs-dynamic-masks-indoor-video010
```

---

## 10. Key Technical Notes

### NaN/Inf Safety
The codebase has numerous NaN/inf guards added throughout:
- `train.py` L1035: Skips optimizer step if loss is non-finite
- `train.py` L1100–1108: Clamps/nan_to_num on all parameters after optimizer step
- `gaussian_renderer/__init__.py`: nan_to_num on all deformation outputs before activation
- These are critical for training stability, especially around the coarse-to-fine temporal transition.

### Gradient Flow
- `viewspace_point_tensor_grad` is accumulated across all batch views (L1067)
- Mask-guided densification boosts gradients for Gaussians inside dynamic masks (L1068–1072)
- Motion-prior densification similarly boosts gradients (L1073–1078)
- Deformation network MLP parameters get separate gradient clipping at `max_norm=10.0` (L1065)

### Memory Management
- Lazy image loading (`viewpoint_cam.load_image()`) to avoid OOM
- `torch.cuda.empty_cache()` called every 50 iterations and after pruning
- `max_points` hard cap with `enforce_max_points()` to stay under GPU memory

### Seed Point Insertion Timing
Seed points from masks are inserted AFTER all per-iteration rendering (densification_stats, max_radii2D) to avoid shape mismatches. This was a notable bug fixed in this fork.

### Layer Rendering with Static Background
When `--layer_static_background` is active, the renderer blends deformed parameters toward original (undeformed) values weighted by `1 - fg_prob`. This keeps static background Gaussians frozen while foreground objects deform normally. It kicks in after `layer_static_background_min_iter` (default 5000).

### Component-Normalized Loss
`component_normalized_dynamic_loss()` in `train.py` uses `scipy.ndimage.label()` to find connected components in each dynamic mask, then averages per-component error means. This prevents tiny objects (tens of pixels) from being drowned out by large background regions in the loss.

---

## 11. Conda Environment

Name: `ed3dgs-stereo`
Key packages: `torch==1.13.1`, `torchvision==0.14.1`, `mmcv==1.6.0`, `lpips`, `plyfile`, `open3d`, `kornia`, `imageio`, `natsort`, and the two CUDA submodules installed via `pip install -e .`

Additional optional packages for metrics: `pyiqa` (for MUSIQ IQ-Score), `scipy` (for connected-component analysis in loss computation).

---

## 12. Upstream vs. Fork Differences Summary

| Area | Upstream E-D3DGS | This Fork |
|------|-----------------|-----------|
| Rendering | Monocular only | Mono + synthetic stereo pairs |
| Supervision | L1 + SSIM | L1 + SSIM + dynamic mask weighted + component-normalized + motion prior + depth + stereo consistency |
| Densification | Gradient-based only | Gradient-based + mask-guided boost + motion-prior boost |
| Pruning | Opacity-based | Opacity-based with optional dynamic mask protection |
| Point seeding | None | Dynamic mask unprojection (monocular + stereo triangulation) |
| Layer model | None | Learned FG/BG probability with layer-separated rendering |
| Background | All Gaussians deform | Optional static background locking |
| Frame sampling | Random + by-error | Random + by-error + dynamic frame oversampling + motion-prior oversampling |
| NaN safety | Minimal | Extensive nan_to_num/clamp guards throughout |
