#!/bin/bash
# E-D3DGS train + mono render + stereo render for SK indoor scenes.
#
# This repository no longer supports dynamic masks, DA3 depth supervision,
# mask seed points, or layered foreground/background training. The retained
# optional sidecar signal is motion priors.
#
# Run from the repository root:
#   bash run_ed3dgs_dynamic_masks_train_render_stereo_interactive.sh video016
#
# Common overrides:
#   GENERATE_MOTION_PRIORS=1 USE_MOTION_PRIORS=1 MOTION_PRIOR_METHOD=raft_torchvision MOTION_PRIOR_DEVICE=cuda bash run_ed3dgs_dynamic_masks_train_render_stereo_interactive.sh video016
#   RUN_TRAIN=0 RUN_RENDER=0 RUN_STEREO=1 IPD=1.4 bash run_ed3dgs_dynamic_masks_train_render_stereo_interactive.sh video016

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-${SCRIPT_DIR}}"

SCENES=(video013)
if [ $# -gt 0 ]; then
    SCENES=("$@")
fi

DATASET_ROOT="${DATASET_ROOT:-/rds/general/user/ka1525/home/ephemeral-1/datasets/SK/indoor}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ed3dgs-no-masks-stereo-reg-raft-motion-priors-indoor}"
ENV_NAME="${ENV_NAME:-ed3dgs-stereo}"

# Pipeline switches.
GENERATE_MOTION_PRIORS="${GENERATE_MOTION_PRIORS:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_STEREO="${RUN_STEREO:-1}"
RUN_METRICS="${RUN_METRICS:-0}"
COMPUTE_IQ="${COMPUTE_IQ:-0}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

# Training/render parameters.
ITERATIONS="${ITERATIONS:-30000}"
TOTAL_FRAMES="${TOTAL_FRAMES:-300}"
MAXTIME="${MAXTIME:-${TOTAL_FRAMES}}"
MAX_POINTS="${MAX_POINTS:-350000}"
C2F_TEMPORAL_ITER="${C2F_TEMPORAL_ITER:-30000}"
RESOLUTION_SCALE="${RESOLUTION_SCALE:-2}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
USE_CONFIG="${USE_CONFIG:-0}"
CONFIG_FILE="${CONFIG_FILE:-arguments/dynerf/default.py}"

# Motion-prior parameters.
USE_MOTION_PRIORS="${USE_MOTION_PRIORS:-0}"
MOTION_PRIOR_DIR="${MOTION_PRIOR_DIR:-motion_priors}"
MOTION_PRIOR_METHOD="${MOTION_PRIOR_METHOD:-farneback}"
MOTION_PRIOR_DEVICE="${MOTION_PRIOR_DEVICE:-cuda}"
RAFT_TORCH_HOME="${RAFT_TORCH_HOME:-/rds/general/user/ka1525/home/ephemeral-1/raft_torchvision}"
MOTION_PRIOR_WEIGHT="${MOTION_PRIOR_WEIGHT:-0.0}"
MOTION_PRIOR_THRESHOLD="${MOTION_PRIOR_THRESHOLD:-0.35}"
MOTION_PRIOR_MIN_AREA="${MOTION_PRIOR_MIN_AREA:-0.00005}"
MOTION_PRIOR_MAX_AREA="${MOTION_PRIOR_MAX_AREA:-0.05}"
MOTION_PRIOR_FRAME_SAMPLE_PROB="${MOTION_PRIOR_FRAME_SAMPLE_PROB:-0.0}"
MOTION_PRIOR_FRAME_SAMPLE_MIN_AREA="${MOTION_PRIOR_FRAME_SAMPLE_MIN_AREA:-0.00005}"
USE_MOTION_PRIOR_DENSIFICATION="${USE_MOTION_PRIOR_DENSIFICATION:-0}"
MOTION_PRIOR_DENSIFY_GRAD_BOOST="${MOTION_PRIOR_DENSIFY_GRAD_BOOST:-2.0}"
MOTION_PRIOR_PERCENTILE="${MOTION_PRIOR_PERCENTILE:-98.5}"
MOTION_PRIOR_MAX_COMPONENT_AREA="${MOTION_PRIOR_MAX_COMPONENT_AREA:-0.05}"
MOTION_PRIOR_MAX_COMPONENTS="${MOTION_PRIOR_MAX_COMPONENTS:-8}"

# Synthetic stereo-view consistency training.
USE_SYNTHETIC_STEREO_CONSISTENCY="${USE_SYNTHETIC_STEREO_CONSISTENCY:-0}"
LAMBDA_STEREO_CONSISTENCY="${LAMBDA_STEREO_CONSISTENCY:-0.05}"
STEREO_TRAIN_BASELINE="${STEREO_TRAIN_BASELINE:-0.3}"
STEREO_OCCLUSION_TOLERANCE="${STEREO_OCCLUSION_TOLERANCE:-0.01}"

# Mono render parameters.
RENDER_SKIP_TRAIN="${RENDER_SKIP_TRAIN:-0}"
RENDER_SKIP_TEST="${RENDER_SKIP_TEST:-1}"
RENDER_SKIP_VIDEO="${RENDER_SKIP_VIDEO:-1}"

# Stereo render parameters.
IPD="${IPD:-0.12}"
CONVERGENCE="${CONVERGENCE:-0.0}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-side_by_side}"
FPS="${FPS:-30.0}"
STEREO_SKIP_TRAIN="${STEREO_SKIP_TRAIN:-0}"
STEREO_SKIP_TEST="${STEREO_SKIP_TEST:-0}"
STEREO_SKIP_VIDEO="${STEREO_SKIP_VIDEO:-0}"

require_cuda() {
    python - <<'PY'
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print(f"CUDA available: {torch.cuda.is_available()} | device_count: {torch.cuda.device_count()}")
sys.exit(0 if ok else 1)
PY
}

normalize_cuda_visible_devices() {
    local cvd="${CUDA_VISIBLE_DEVICES:-}"
    if [ -n "${GPU_ID}" ]; then
        cvd="${GPU_ID}"
    fi

    if [ -z "${cvd}" ]; then
        return 0
    fi

    if [[ "${cvd}" == *GPU-* ]]; then
        local mapped=()
        local token idx
        local tokens
        IFS=',' read -r -a tokens <<< "${cvd}"
        for token in "${tokens[@]}"; do
            token="${token//[[:space:]]/}"
            if [[ "${token}" == GPU-* ]]; then
                idx="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' -v u="${token}" '$2==u {print $1; exit}')"
                if [ -z "${idx}" ]; then
                    echo ">>> Failed to map CUDA_VISIBLE_DEVICES UUID ${token} to GPU index."
                    return 1
                fi
                mapped+=("${idx}")
            else
                mapped+=("${token}")
            fi
        done
        export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${mapped[*]}")"
    else
        export CUDA_VISIBLE_DEVICES="${cvd}"
    fi
}

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load tools/prod
    module load miniforge/3
    module load CUDA/11.7.0
    module load FFmpeg/6.0-GCCcore-12.3.0 || true
fi

if [ -f "${HOME}/miniforge3/bin/conda" ]; then
    eval "$("${HOME}/miniforge3/bin/conda" shell.bash hook)"
fi
set +u
conda activate "${ENV_NAME}"
set -u

export CUDA_HOME="${EBROOTCUDA:-/sw-eb/software/CUDA/11.7.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

if ! normalize_cuda_visible_devices; then
    exit 1
fi

cd "${REPO}"

echo "--- GPU status ---"
echo "Repository: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi
if ! require_cuda; then
    echo ">>> No visible CUDA device. Start a GPU session and re-run."
    exit 1
fi
echo "------------------"

if [ "${RUN_STEREO}" = "1" ] && ! command -v ffmpeg >/dev/null 2>&1; then
    echo ">>> ffmpeg is not available in PATH. Stereo video encoding may fail."
fi

for SCENE in "${SCENES[@]}"; do
    DATA_PATH="${DATASET_ROOT}/${SCENE}"
    OUTPUT_PATH="output/${OUTPUT_PREFIX}-${SCENE}"

    echo ""
    echo "=========================================="
    echo "Scene: ${SCENE}"
    echo "Dataset: ${DATA_PATH}"
    echo "Output:  $(pwd)/${OUTPUT_PATH}"
    echo "Motion priors: use=${USE_MOTION_PRIORS} generate=${GENERATE_MOTION_PRIORS} dir=${MOTION_PRIOR_DIR} method=${MOTION_PRIOR_METHOD} weight=${MOTION_PRIOR_WEIGHT} oversample=${MOTION_PRIOR_FRAME_SAMPLE_PROB} densify=${USE_MOTION_PRIOR_DENSIFICATION}"
    echo "Stereo consistency: use=${USE_SYNTHETIC_STEREO_CONSISTENCY} lambda=${LAMBDA_STEREO_CONSISTENCY} baseline=${STEREO_TRAIN_BASELINE}"
    echo "=========================================="

    if [ ! -d "${DATA_PATH}/images" ]; then
        echo ">>> Missing ${DATA_PATH}/images, skipping."
        continue
    fi

    if [ ! -d "${DATA_PATH}/ns_output/colmap/sparse/0" ] && [ ! -d "${DATA_PATH}/ns_output/colmap/sparse_work/0" ]; then
        echo ">>> Missing sparse model under ns_output/colmap for ${SCENE}, skipping."
        continue
    fi

    if [ "${GENERATE_MOTION_PRIORS}" = "1" ]; then
        echo ">>> [GENERATE MOTION PRIORS] ${SCENE}"
        python script/generate_motion_priors.py \
            --source "${DATA_PATH}" \
            --images images \
            --output "${MOTION_PRIOR_DIR}" \
            --method "${MOTION_PRIOR_METHOD}" \
            --device "${MOTION_PRIOR_DEVICE}" \
            --torch_home "${RAFT_TORCH_HOME}" \
            --percentile "${MOTION_PRIOR_PERCENTILE}" \
            --threshold "${MOTION_PRIOR_THRESHOLD}" \
            --max_area "${MOTION_PRIOR_MAX_COMPONENT_AREA}" \
            --max_components "${MOTION_PRIOR_MAX_COMPONENTS}" \
            --overlay_dir "motion_prior_overlays_${MOTION_PRIOR_DIR}"
    fi

    if [ "${USE_MOTION_PRIORS}" = "1" ] && [ ! -d "${DATA_PATH}/${MOTION_PRIOR_DIR}" ]; then
        echo ">>> Motion-prior directory missing: ${DATA_PATH}/${MOTION_PRIOR_DIR}"
        echo ">>> Set GENERATE_MOTION_PRIORS=1 or MOTION_PRIOR_DIR to an existing motion-prior folder."
        exit 1
    fi

    if [ "${CLEAN_OUTPUT}" = "1" ] && [ -d "${OUTPUT_PATH}" ]; then
        echo ">>> CLEAN_OUTPUT=1, removing ${OUTPUT_PATH}"
        rm -rf "${OUTPUT_PATH}"
    fi

    if [ "${RUN_TRAIN}" = "1" ]; then
        echo ">>> [TRAINING] ${SCENE}"
        TRAIN_ARGS=(
            -s "${DATA_PATH}"
            --loader dynerf
            --images images
            --model_path "${OUTPUT_PATH}"
            --expname "dynerf/${SCENE}"
            --iterations "${ITERATIONS}"
            --maxtime "${MAXTIME}"
            --total_num_frames "${TOTAL_FRAMES}"
            --max_points "${MAX_POINTS}"
            --c2f_temporal_iter "${C2F_TEMPORAL_ITER}"
            -r "${RESOLUTION_SCALE}"
        )

        if [ "${USE_MOTION_PRIORS}" = "1" ]; then
            TRAIN_ARGS+=(
                --use_motion_priors
                --motion_prior_dir "${MOTION_PRIOR_DIR}"
                --motion_prior_loss_weight "${MOTION_PRIOR_WEIGHT}"
                --motion_prior_threshold "${MOTION_PRIOR_THRESHOLD}"
                --motion_prior_min_area "${MOTION_PRIOR_MIN_AREA}"
                --motion_prior_max_area "${MOTION_PRIOR_MAX_AREA}"
                --motion_prior_frame_sample_prob "${MOTION_PRIOR_FRAME_SAMPLE_PROB}"
                --motion_prior_frame_sample_min_area "${MOTION_PRIOR_FRAME_SAMPLE_MIN_AREA}"
                --motion_prior_densify_grad_boost "${MOTION_PRIOR_DENSIFY_GRAD_BOOST}"
            )
            if [ "${USE_MOTION_PRIOR_DENSIFICATION}" = "1" ]; then
                TRAIN_ARGS+=(--use_motion_prior_densification)
            fi
        fi

        if [ "${USE_SYNTHETIC_STEREO_CONSISTENCY}" = "1" ]; then
            TRAIN_ARGS+=(
                --lambda_stereo_consistency "${LAMBDA_STEREO_CONSISTENCY}"
                --stereo_baseline "${STEREO_TRAIN_BASELINE}"
                --stereo_occlusion_tolerance "${STEREO_OCCLUSION_TOLERANCE}"
            )
        fi

        if [ "${USE_CONFIG}" = "1" ]; then
            TRAIN_ARGS+=(--configs "${CONFIG_FILE}")
        fi

        python train.py "${TRAIN_ARGS[@]}"
    fi

    if [ "${RUN_RENDER}" = "1" ]; then
        echo ">>> [MONO RENDER] ${SCENE}"
        RENDER_ARGS=(--model_path "${OUTPUT_PATH}")
        if [ "${USE_CONFIG}" = "1" ]; then
            RENDER_ARGS+=(--configs "${CONFIG_FILE}")
        fi
        if [ "${RENDER_SKIP_TRAIN}" = "1" ]; then
            RENDER_ARGS+=(--skip_train)
        fi
        if [ "${RENDER_SKIP_TEST}" = "1" ]; then
            RENDER_ARGS+=(--skip_test)
        fi
        if [ "${RENDER_SKIP_VIDEO}" = "1" ]; then
            RENDER_ARGS+=(--skip_video)
        fi
        python render.py "${RENDER_ARGS[@]}"
    fi

    if [ "${RUN_STEREO}" = "1" ]; then
        echo ">>> [STEREO RENDER] ${SCENE}"
        STEREO_ARGS=(
            --model_path "${OUTPUT_PATH}"
            --ipd "${IPD}"
            --convergence_distance "${CONVERGENCE}"
            --output_format "${OUTPUT_FORMAT}"
            --gt_source_path "${DATA_PATH}"
            --fps "${FPS}"
        )
        if [ "${USE_CONFIG}" = "1" ]; then
            STEREO_ARGS+=(--configs "${CONFIG_FILE}")
        fi
        if [ "${STEREO_SKIP_TRAIN}" = "1" ]; then
            STEREO_ARGS+=(--skip_train)
        fi
        if [ "${STEREO_SKIP_TEST}" = "1" ]; then
            STEREO_ARGS+=(--skip_test)
        fi
        if [ "${STEREO_SKIP_VIDEO}" = "1" ]; then
            STEREO_ARGS+=(--skip_video)
        fi
        python render_stereo.py "${STEREO_ARGS[@]}"

        LATEST_STEREO_DIR="$(find "${OUTPUT_PATH}/train" -maxdepth 1 -mindepth 1 -type d -name 'stereo_*' 2>/dev/null | sort -V | tail -n 1 || true)"
        if [ -n "${LATEST_STEREO_DIR}" ]; then
            mkdir -p "${OUTPUT_PATH}/all_frames"
            ln -sfn "${LATEST_STEREO_DIR}/renders" "${OUTPUT_PATH}/all_frames/stereo_renders"
            echo ">>> All-frame stereo renders: ${OUTPUT_PATH}/all_frames/stereo_renders"
        fi
    fi

    if [ "${RUN_METRICS}" = "1" ]; then
        echo ">>> [METRICS] ${SCENE}"
        METRIC_ARGS=(--model_paths "${OUTPUT_PATH}")
        if [ "${COMPUTE_IQ}" = "1" ]; then
            METRIC_ARGS+=(--compute_iq)
        fi
        python metrics.py "${METRIC_ARGS[@]}"
    fi

    echo ">>> Done: ${SCENE} -> ${OUTPUT_PATH}"
done

echo ""
echo "All scenes complete."
