#!/usr/bin/env bash

# Simple Docker DDP launcher for BEVNeXt-SAM2 (2x GPUs by default)
# - Uses torchrun for proper multi-process DistributedDataParallel
# - Designed for remote servers with Docker and 2× 2080 Ti GPUs

set -euo pipefail

# Defaults
NUM_GPUS=${NUM_GPUS:-2}
GPU_IDS=""
BATCH_SIZE=${BATCH_SIZE:-4}          # per-GPU
EPOCHS=${EPOCHS:-50}
DATA_PATH=${DATA_PATH:-/data/nuscenes}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/train_ddp}
IMAGE_NAME=${IMAGE_NAME:-bevnext-sam2}
CONTAINER_NAME=${CONTAINER_NAME:-bevnext-sam2-ddp}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_FILE=""
RESUME_CKPT=""
MIXED_PRECISION=1
DETACH=0

print_help() {
  cat <<EOF
BEVNeXt-SAM2: Docker DDP training launcher

Usage: $0 [options]

Options:
  --data-path PATH          Host path to nuScenes dataset (default: /data/nuscenes)
  --num-gpus N              Number of GPUs to use (default: 2)
  --gpus LIST               Explicit GPU IDs (e.g., "0,1"). Overrides --num-gpus
  --batch-size N            Per-GPU batch size (default: 4)
  --epochs N                Number of epochs (default: 50)
  --config PATH             Config file (relative to repo root or absolute)
  --resume PATH             Checkpoint to resume from
  --output-dir PATH         Output dir on host (default: outputs/train_ddp)
  --image NAME              Docker image name (default: bevnext-sam2)
  --container-name NAME     Container name (default: bevnext-sam2-ddp)
  --port N                  DDP master port (default: 29500)
  --no-mixed-precision      Disable AMP mixed precision
  --detach                  Run container in background (docker -d)
  -h | --help               Show this help

Examples:
  # Build image once, then launch 2-GPU DDP training
  ./scripts/run.sh build-fast --gpu
  ./scripts/train_ddp_docker.sh --data-path /data/nuscenes --num-gpus 2 --batch-size 4 --epochs 50

  # Explicit GPU IDs and resume
  ./scripts/train_ddp_docker.sh --gpus 0,1 --resume outputs/checkpoints/checkpoint_latest.pth
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="$2"; shift 2;;
    --num-gpus) NUM_GPUS="$2"; shift 2;;
    --gpus) GPU_IDS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --config) CONFIG_FILE="$2"; shift 2;;
    --resume) RESUME_CKPT="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --image) IMAGE_NAME="$2"; shift 2;;
    --container-name) CONTAINER_NAME="$2"; shift 2;;
    --port) MASTER_PORT="$2"; shift 2;;
    --no-mixed-precision) MIXED_PRECISION=0; shift;;
    --detach) DETACH=1; shift;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown option: $1"; print_help; exit 1;;
  esac
done

# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Validate GPUs on host
if command -v nvidia-smi >/dev/null 2>&1; then
  AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
  echo "nvidia-smi not found; ensure NVIDIA drivers and nvidia-container-runtime are installed." >&2
  exit 1
fi

if [[ -z "$GPU_IDS" ]]; then
  if [[ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]]; then
    echo "Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available" >&2
    exit 1
  fi
  # Generate GPU IDs 0..N-1
  GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
else
  # Derive NUM_GPUS from explicit IDs
  NUM_GPUS=$(echo "$GPU_IDS" | awk -F, '{print NF}')
fi

# Normalize host paths
ABS_PROJECT_PATH="$(realpath "$PROJECT_ROOT")"
if [[ -d "$DATA_PATH" ]]; then
  ABS_DATA_PATH="$(realpath "$DATA_PATH")"
else
  echo "Dataset path does not exist: $DATA_PATH" >&2
  exit 1
fi
ABS_OUTPUTS_PATH="$(mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR" && realpath "$PROJECT_ROOT/$OUTPUT_DIR")"
ABS_LOGS_PATH="$(mkdir -p "$PROJECT_ROOT/logs" && realpath "$PROJECT_ROOT/logs")"

echo "======================================"
echo "BEVNeXt-SAM2 Docker DDP Training"
echo "======================================"
echo "GPUs:            $GPU_IDS ($NUM_GPUS)"
echo "Data root:       $ABS_DATA_PATH"
echo "Output dir:      $ABS_OUTPUTS_PATH"
echo "Epochs:          $EPOCHS"
echo "Per-GPU batch:   $BATCH_SIZE"
echo "Mixed precision: $([[ $MIXED_PRECISION -eq 1 ]] && echo on || echo off)"
echo "Image:           $IMAGE_NAME"
echo "Master port:     $MASTER_PORT"
echo "Container:       $CONTAINER_NAME"
[[ -n "$CONFIG_FILE" ]] && echo "Config:          $CONFIG_FILE"
[[ -n "$RESUME_CKPT" ]] && echo "Resume:          $RESUME_CKPT"
echo "======================================"

# Build torchrun command
TRAIN_ARGS=(
  "--data-root" "/workspace/data/nuscenes"
  "--epochs" "$EPOCHS"
  "--batch-size" "$BATCH_SIZE"
  "--gpus" "$GPU_IDS"
  "--distributed"
  "--lr-scaling" "linear"
)
[[ $MIXED_PRECISION -eq 1 ]] && TRAIN_ARGS+=("--mixed-precision")
[[ -n "$CONFIG_FILE" ]] && TRAIN_ARGS+=("--config" "/workspace/bevnext-sam2/$CONFIG_FILE")
[[ -n "$RESUME_CKPT" ]] && TRAIN_ARGS+=("--resume" "/workspace/bevnext-sam2/$RESUME_CKPT")

INNER_CMD=$(cat <<EOF
set -euo pipefail
export NUSCENES_DATA_ROOT=/workspace/data/nuscenes
export PYTHONPATH=/workspace/bevnext-sam2
export OMP_NUM_THREADS=1
export TORCH_CUDNN_BENCHMARK=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0

echo "Launching torchrun with $NUM_GPUS processes..."
torchrun --nproc_per_node=$NUM_GPUS --master_addr=127.0.0.1 --master_port=$MASTER_PORT \
  training/train_bevnext_sam2_nuscenes.py ${TRAIN_ARGS[*]}
EOF
)

# Docker run command
DOCKER_CMD=(
  docker run --rm
  --name "$CONTAINER_NAME"
  --gpus "device=$GPU_IDS"
  --shm-size=8g
  --ulimit memlock=-1 --ulimit stack=67108864
  -e NUSCENES_DATA_ROOT=/workspace/data/nuscenes
  -e PYTHONPATH=/workspace/bevnext-sam2
  -v "$ABS_PROJECT_PATH":/workspace/bevnext-sam2
  -v "$ABS_DATA_PATH":/workspace/data/nuscenes:ro
  -v "$ABS_OUTPUTS_PATH":/workspace/outputs
  -v "$ABS_LOGS_PATH":/workspace/logs
  -w /workspace/bevnext-sam2
  "$IMAGE_NAME"
  bash -lc "$INNER_CMD"
)

if [[ $DETACH -eq 1 ]]; then
  DOCKER_CMD=(docker run -d "${DOCKER_CMD[@]:2}")
fi

echo "Running: ${DOCKER_CMD[*]}"
"${DOCKER_CMD[@]}"

echo "Done. To monitor logs:"
if [[ $DETACH -eq 1 ]]; then
  echo "  docker logs -f $CONTAINER_NAME"
else
  echo "  scripts/monitor_training.sh"
fi


