#!/bin/bash
"""
Docker Multi-GPU Training Launch Script for BEVNeXt-SAM2
Supports both DataParallel and DistributedDataParallel training in Docker containers
"""

set -e

# Configuration
PROJECT_NAME="bevnext-sam2"
IMAGE_NAME="bevnext-sam2"
CONTAINER_NAME="bevnext-sam2-multigpu"
DEFAULT_DATA_PATH="/data/nuscenes"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ABS_PROJECT_PATH="$(realpath "$PROJECT_ROOT")"
ABS_OUTPUTS_PATH="$(realpath "$PROJECT_ROOT/outputs" 2>/dev/null || echo "$PROJECT_ROOT/outputs")"
ABS_LOGS_PATH="$(realpath "$PROJECT_ROOT/logs" 2>/dev/null || echo "$PROJECT_ROOT/logs")"

# Default configuration
NUM_GPUS=2
BATCH_SIZE=4  # Per GPU batch size
EPOCHS=50
DATA_ROOT="data/nuscenes"
OUTPUT_DIR="outputs/docker_multi_gpu"
CONFIG_FILE=""
RESUME_CHECKPOINT=""
GRADIENT_ACCUMULATION=1
LR_SCALING="linear"
MIXED_PRECISION=true
TRAINING_SCRIPT="train_bevnext_sam2_nuscenes.py"
DISTRIBUTED_MODE="auto"  # auto, dp, ddp
DOCKER_OPTS="--rm -it --user $(id -u):$(id -g)"
USE_GPU=true
DRY_RUN=false

print_help() {
    echo -e "${BLUE}BEVNeXt-SAM2 Docker Multi-GPU Training Launcher${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Multi-GPU Configuration:"
    echo "  --num-gpus NUM             Number of GPUs to use (default: 2)"
    echo "  --batch-size SIZE          Per-GPU batch size (default: 4)"
    echo "  --epochs NUM               Number of training epochs (default: 50)"
    echo "  --data-root PATH           Path to nuScenes dataset (default: data/nuscenes)"
    echo "  --output-dir PATH          Output directory (default: outputs/docker_multi_gpu)"
    echo "  --config PATH              Path to config file (optional)"
    echo "  --resume PATH              Path to checkpoint to resume from (optional)"
    echo "  --gradient-accumulation N  Gradient accumulation steps (default: 1)"
    echo "  --lr-scaling RULE          Learning rate scaling rule: linear|sqrt|none (default: linear)"
    echo "  --no-mixed-precision       Disable mixed precision training"
    echo "  --basic-training           Use basic training script instead of nuScenes"
    echo ""
    echo "Docker Configuration:"
    echo "  --distributed MODE         Distributed mode: auto|dp|ddp (default: auto)"
    echo "  --docker-image NAME        Docker image name (default: bevnext-sam2)"
    echo "  --container-name NAME      Container name (default: bevnext-sam2-multigpu)"
    echo "  --no-gpu                   Disable GPU support (CPU only)"
    echo "  --dry-run                  Show commands without executing"
    echo ""
    echo "Examples:"
    echo "  $0 --num-gpus 2 --batch-size 4 --epochs 50"
    echo "  $0 --distributed ddp --gradient-accumulation 2 --lr-scaling linear"
    echo "  $0 --basic-training --num-gpus 2 --batch-size 6"
    echo ""
    echo "Docker Commands:"
    echo "  # Build container first:"
    echo "  ./scripts/run.sh build-fast --gpu"
    echo ""
    echo "  # Then run multi-GPU training:"
    echo "  $0 --num-gpus 2 --batch-size 4"
}

check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

setup_gpu() {
    if check_gpu; then
        echo -e "${GREEN}GPU detected - enabling GPU support${NC}"
        DOCKER_OPTS="$DOCKER_OPTS --gpus all"
        export NVIDIA_VISIBLE_DEVICES=all
    else
        echo -e "${YELLOW}No GPU detected - running on CPU${NC}"
    fi
}

remove_container() {
    local container_name="$1"
    if docker ps -a --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        echo -e "${YELLOW}Removing existing container: ${container_name}${NC}"
        docker rm -f "$container_name" >/dev/null 2>&1 || true
    fi
}

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
    --distributed)
      DISTRIBUTED_MODE="$2"
      shift 2
      ;;
    --docker-image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --no-gpu)
      USE_GPU=false
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      print_help
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option $1${NC}"
      print_help
      exit 1
      ;;
  esac
done

# Validate GPU count
if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
    echo -e "${RED}Error: Invalid number of GPUs: $NUM_GPUS${NC}"
    exit 1
fi

# Check if GPUs are available
if [[ "$USE_GPU" != "false" ]]; then
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if [[ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]]; then
        echo -e "${RED}Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available${NC}"
        exit 1
    fi
    setup_gpu
fi

# Generate GPU IDs string (0,1,2,...)
GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))

# Determine distributed mode
if [[ "$DISTRIBUTED_MODE" == "auto" ]]; then
    if [[ "$NUM_GPUS" -eq 1 ]]; then
        DISTRIBUTED_MODE="none"
    elif [[ "$NUM_GPUS" -le 4 ]]; then
        DISTRIBUTED_MODE="ddp"
    else
        DISTRIBUTED_MODE="ddp"
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert data path to absolute path for Docker volume mounting
if [[ -d "$DATA_ROOT" ]]; then
    ABS_DATA_PATH="$(realpath "$DATA_ROOT")"
    echo -e "${GREEN}Using dataset: $ABS_DATA_PATH${NC}"
else
    ABS_DATA_PATH="$DATA_ROOT"
    echo -e "${YELLOW}Dataset path not found locally: $DATA_ROOT${NC}"
    echo -e "${YELLOW}Will attempt to use as-is (may be a volume or remote path)${NC}"
fi

echo "======================================"
echo "BEVNeXt-SAM2 Docker Multi-GPU Training"
echo "======================================"
echo "Configuration:"
echo "  - GPUs: $NUM_GPUS ($GPU_IDS)"
echo "  - Batch size per GPU: $BATCH_SIZE"
echo "  - Total effective batch size: $((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "  - Epochs: $EPOCHS"
echo "  - Data root: $DATA_ROOT"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Training script: $TRAINING_SCRIPT"
echo "  - Distributed mode: $DISTRIBUTED_MODE"
echo "  - Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "  - LR scaling: $LR_SCALING"
echo "  - Mixed precision: $MIXED_PRECISION"
echo "  - Docker image: $IMAGE_NAME"
if [[ -n "$CONFIG_FILE" ]]; then
    echo "  - Config file: $CONFIG_FILE"
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "  - Resume from: $RESUME_CHECKPOINT"
fi
echo "======================================"

# Build training command arguments
TRAIN_ARGS=(
    "--data-root" "/workspace/data/nuscenes"
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
    TRAIN_ARGS+=("--config" "/workspace/bevnext-sam2/$CONFIG_FILE")
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=("--resume" "/workspace/bevnext-sam2/$RESUME_CHECKPOINT")
fi

# Function to run DataParallel training in Docker
run_docker_dataparallel() {
    echo -e "${GREEN}Starting DataParallel training in Docker...${NC}"
    
    remove_container "$CONTAINER_NAME"
    
    # Build Docker command
    DOCKER_CMD="docker run $DOCKER_OPTS"
    DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
    DOCKER_CMD="$DOCKER_CMD -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD -v $ABS_DATA_PATH:/workspace/data/nuscenes:ro"
    DOCKER_CMD="$DOCKER_CMD -v $ABS_OUTPUTS_PATH:/workspace/outputs"
    DOCKER_CMD="$DOCKER_CMD -v $ABS_LOGS_PATH:/workspace/logs"
    DOCKER_CMD="$DOCKER_CMD -w /workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD -e NUSCENES_DATA_ROOT=/workspace/data/nuscenes"
    DOCKER_CMD="$DOCKER_CMD -e PYTHONPATH=/workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
    
    # Build training command
    TRAIN_CMD="python training/$TRAINING_SCRIPT ${TRAIN_ARGS[*]}"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "$DOCKER_CMD bash -c \"$TRAIN_CMD\""
    else
        echo -e "${GREEN}🚀 Starting DataParallel training in Docker...${NC}"
        eval "$DOCKER_CMD bash -c \"$TRAIN_CMD\""
    fi
}

# Function to run DistributedDataParallel training in Docker
run_docker_distributed() {
    echo -e "${GREEN}Starting DistributedDataParallel training in Docker...${NC}"
    
    remove_container "$CONTAINER_NAME"
    
    # Build Docker command
    DOCKER_CMD="docker run $DOCKER_OPTS"
    DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
    DOCKER_CMD="$DOCKER_CMD -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD -v $ABS_DATA_PATH:/workspace/data/nuscenes:ro"
    DOCKER_CMD="$DOCKER_CMD -v $ABS_OUTPUTS_PATH:/workspace/outputs"
    DOCKER_CMD="$DOCKER_CMD -v $ABS_LOGS_PATH:/workspace/logs"
    DOCKER_CMD="$DOCKER_CMD -w /workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD -e NUSCENES_DATA_ROOT=/workspace/data/nuscenes"
    DOCKER_CMD="$DOCKER_CMD -e PYTHONPATH=/workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
    
    # Add distributed flag
    TRAIN_ARGS+=("--distributed")
    
    # Build training command
    TRAIN_CMD="python training/$TRAINING_SCRIPT ${TRAIN_ARGS[*]}"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "$DOCKER_CMD bash -c \"$TRAIN_CMD\""
    else
        echo -e "${GREEN}🚀 Starting DistributedDataParallel training in Docker...${NC}"
        eval "$DOCKER_CMD bash -c \"$TRAIN_CMD\""
    fi
}

# Main execution
if [[ "$NUM_GPUS" -eq 1 ]]; then
    echo -e "${YELLOW}Single GPU detected, running without parallelization${NC}"
    run_docker_dataparallel
elif [[ "$DISTRIBUTED_MODE" == "dp" ]]; then
    echo -e "${GREEN}Running DataParallel training in Docker${NC}"
    run_docker_dataparallel
elif [[ "$DISTRIBUTED_MODE" == "ddp" ]]; then
    echo -e "${GREEN}Running DistributedDataParallel training in Docker (recommended for 2-4 GPUs)${NC}"
    run_docker_distributed
else
    echo -e "${GREEN}Running DistributedDataParallel training in Docker (recommended for 2-4 GPUs)${NC}"
    run_docker_distributed
fi

echo -e "${GREEN}Docker multi-GPU training completed!${NC}" 