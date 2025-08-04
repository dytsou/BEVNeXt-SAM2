#!/bin/bash
"""
Distributed Training Launch Script for BEVNeXt-SAM2
Supports both DataParallel and DistributedDataParallel training modes
"""

set -e

# Default configuration
NUM_GPUS=2
BATCH_SIZE=4  # Per GPU batch size
EPOCHS=50
DATA_ROOT="data/nuscenes"
OUTPUT_DIR="outputs/distributed_training"
CONFIG_FILE=""
RESUME_CHECKPOINT=""
GRADIENT_ACCUMULATION=1
LR_SCALING="linear"
MIXED_PRECISION=true
TRAINING_SCRIPT="train_bevnext_sam2_nuscenes.py"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --resume)
      RESUME_CHECKPOINT="$2"
      shift 2
      ;;
    --gradient-accumulation)
      GRADIENT_ACCUMULATION="$2"
      shift 2
      ;;
    --lr-scaling)
      LR_SCALING="$2"
      shift 2
      ;;
    --no-mixed-precision)
      MIXED_PRECISION=false
      shift
      ;;
    --basic-training)
      TRAINING_SCRIPT="train_bevnext_sam2.py"
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --num-gpus NUM             Number of GPUs to use (default: 2)"
      echo "  --batch-size SIZE          Per-GPU batch size (default: 4)"
      echo "  --epochs NUM               Number of training epochs (default: 50)"
      echo "  --data-root PATH           Path to nuScenes dataset (default: data/nuscenes)"
      echo "  --output-dir PATH          Output directory (default: outputs/distributed_training)"
      echo "  --config PATH              Path to config file (optional)"
      echo "  --resume PATH              Path to checkpoint to resume from (optional)"
      echo "  --gradient-accumulation N  Gradient accumulation steps (default: 1)"
      echo "  --lr-scaling RULE          Learning rate scaling rule: linear|sqrt|none (default: linear)"
      echo "  --no-mixed-precision       Disable mixed precision training"
      echo "  --basic-training           Use basic training script instead of nuScenes"
      echo "  --help                     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Validate GPU count
if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "Error: Invalid number of GPUs: $NUM_GPUS"
  exit 1
fi

# Check if GPUs are available
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]]; then
  echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate GPU IDs string (0,1,2,...)
GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))

echo "======================================"
echo "BEVNeXt-SAM2 Distributed Training"
echo "======================================"
echo "Configuration:"
echo "  - GPUs: $NUM_GPUS ($GPU_IDS)"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Total effective batch size: $((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "  - Epochs: $EPOCHS"
echo "  - Data root: $DATA_ROOT"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Training script: $TRAINING_SCRIPT"
echo "  - Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "  - LR scaling: $LR_SCALING"
echo "  - Mixed precision: $MIXED_PRECISION"
if [[ -n "$CONFIG_FILE" ]]; then
  echo "  - Config file: $CONFIG_FILE"
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  echo "  - Resume from: $RESUME_CHECKPOINT"
fi
echo "======================================"

# Build training command arguments
TRAIN_ARGS=(
  "--data-root" "$DATA_ROOT"
  "--epochs" "$EPOCHS"
  "--batch-size" "$BATCH_SIZE"
  "--gpus" "$GPU_IDS"
  "--gradient-accumulation" "$GRADIENT_ACCUMULATION"
  "--lr-scaling" "$LR_SCALING"
)

if [[ "$MIXED_PRECISION" == "true" ]]; then
  TRAIN_ARGS+=("--mixed-precision")
fi

if [[ -n "$CONFIG_FILE" ]]; then
  TRAIN_ARGS+=("--config" "$CONFIG_FILE")
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=("--resume" "$RESUME_CHECKPOINT")
fi

# Function to run DataParallel training (single process, multiple GPUs)
run_dataparallel() {
  echo "Starting DataParallel training..."
  cd "$(dirname "$0")/.."
  
  python "training/$TRAINING_SCRIPT" "${TRAIN_ARGS[@]}"
}

# Function to run DistributedDataParallel training (multiple processes)
run_distributed() {
  echo "Starting DistributedDataParallel training..."
  cd "$(dirname "$0")/.."
  
  # Add distributed flag
  TRAIN_ARGS+=("--distributed")
  
  # Check if torchrun is available (PyTorch 1.10+)
  if command -v torchrun &> /dev/null; then
    echo "Using torchrun for distributed training"
    torchrun \
      --nproc_per_node="$NUM_GPUS" \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=localhost \
      --master_port=29500 \
      "training/$TRAINING_SCRIPT" "${TRAIN_ARGS[@]}"
  else
    # Fallback to python -m torch.distributed.launch
    echo "Using torch.distributed.launch for distributed training"
    python -m torch.distributed.launch \
      --nproc_per_node="$NUM_GPUS" \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=localhost \
      --master_port=29500 \
      "training/$TRAINING_SCRIPT" "${TRAIN_ARGS[@]}"
  fi
}

# Function to run GPU benchmark
run_benchmark() {
  echo "Running GPU benchmark..."
  
  # Test DataParallel
  echo "Testing DataParallel mode..."
  start_time=$(date +%s)
  run_dataparallel || true
  dp_time=$(($(date +%s) - start_time))
  
  # Test DistributedDataParallel
  echo "Testing DistributedDataParallel mode..."
  start_time=$(date +%s)
  run_distributed || true
  ddp_time=$(($(date +%s) - start_time))
  
  echo "======================================"
  echo "Benchmark Results:"
  echo "  - DataParallel time: ${dp_time}s"
  echo "  - DistributedDataParallel time: ${ddp_time}s"
  if [[ $ddp_time -lt $dp_time ]]; then
    speedup=$(echo "scale=2; $dp_time / $ddp_time" | bc)
    echo "  - DDP speedup: ${speedup}x"
  fi
  echo "======================================"
}

# Main execution
if [[ "$NUM_GPUS" -eq 1 ]]; then
  echo "Single GPU detected, running without parallelization"
  run_dataparallel
elif [[ "$NUM_GPUS" -le 4 ]]; then
  echo "Running DistributedDataParallel training (recommended for 2-4 GPUs)"
  run_distributed
else
  echo "Multiple GPUs detected, defaulting to DistributedDataParallel"
  run_distributed
fi

echo "Training completed!"