#!/bin/bash

# BEVNeXt-SAM2 GPU Training Launcher Script
# This script runs training with CUDA support

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  BEVNeXt-SAM2 GPU Training Launcher  ${NC}"
echo -e "${BLUE}======================================${NC}"

# Default parameters
CONFIG_FILE=""  # Use auto-detection by default
RESUME_CHECKPOINT=""
GPU_ID=0
MIXED_PRECISION="--mixed-precision"
USE_AUTO_DETECTION=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            USE_AUTO_DETECTION=false
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --no-mixed-precision)
            MIXED_PRECISION=""
            shift
            ;;
        --force-config)
            USE_AUTO_DETECTION=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config FILE           Path to config file (default: auto-detection)"
            echo "  --resume FILE           Path to checkpoint to resume from"
            echo "  --gpu ID                GPU ID to use (default: 0)"
            echo "  --no-mixed-precision    Disable mixed precision training"
            echo "  --force-config          Force use of config file instead of auto-detection"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if config file exists (only if specified)
if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
if [ "$USE_AUTO_DETECTION" = true ]; then
    echo -e "  Config: Auto-detection (GPU memory-based)"
else
    echo -e "  Config file: $CONFIG_FILE"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo -e "  Resume from: $RESUME_CHECKPOINT"
fi
echo -e "  GPU ID: $GPU_ID"
if [ -n "$MIXED_PRECISION" ]; then
    echo -e "  Mixed precision: Enabled"
else
    echo -e "  Mixed precision: Disabled"
fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p outputs/training_gpu
mkdir -p logs
mkdir -p checkpoints

# Check for NVIDIA Docker runtime
echo -e "${BLUE}Checking GPU support...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    exit 1
fi

if ! docker info 2>/dev/null | grep -q nvidia; then
    echo -e "${RED}❌ NVIDIA Docker runtime not found.${NC}"
    echo -e "Please install nvidia-docker2:"
    echo -e "  sudo apt update"
    echo -e "  sudo apt install -y nvidia-docker2"
    echo -e "  sudo systemctl restart docker"
    exit 1
fi

# GPU information
echo -e "${GREEN}✓ NVIDIA Docker runtime detected${NC}"
echo -e "${YELLOW}Available GPUs:${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s (%s MB total, %s MB free)\n", $1, $2, $3, $4}'

# Verify GPU availability
nvidia-smi -i "$GPU_ID" &>/dev/null || {
    echo -e "${RED}❌ GPU $GPU_ID not available${NC}"
    exit 1
}

echo -e "${GREEN}✓ Using GPU $GPU_ID${NC}"

# Setup Docker options for GPU training
DOCKER_OPTS="--rm -it"
DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"
DOCKER_OPTS="$DOCKER_OPTS --shm-size=8g"  # Larger shared memory for GPU training
DOCKER_OPTS="$DOCKER_OPTS -e NVIDIA_VISIBLE_DEVICES=$GPU_ID"
DOCKER_OPTS="$DOCKER_OPTS -e CUDA_VISIBLE_DEVICES=$GPU_ID"

# GPU-specific environment variables
DOCKER_OPTS="$DOCKER_OPTS -e CUDA_LAUNCH_BLOCKING=1"  # Better error reporting
DOCKER_OPTS="$DOCKER_OPTS -e TORCH_CUDA_ARCH_LIST='6.0;6.1;7.0;7.5;8.0;8.6'"  # Support multiple architectures

# Add volume mounts
DOCKER_OPTS="$DOCKER_OPTS -v $(realpath .):/workspace/host"
DOCKER_OPTS="$DOCKER_OPTS -v $(realpath ./outputs):/workspace/outputs"
DOCKER_OPTS="$DOCKER_OPTS -v $(realpath ./logs):/workspace/logs"
DOCKER_OPTS="$DOCKER_OPTS -v $(realpath ./checkpoints):/workspace/checkpoints"

# Build training command
TRAIN_CMD="python /workspace/host/training/train_bevnext_sam2.py"

# Only add config if specified
if [ -n "$CONFIG_FILE" ]; then
    TRAIN_CMD="$TRAIN_CMD --config /workspace/host/$CONFIG_FILE"
    if [ "$USE_AUTO_DETECTION" = false ]; then
        TRAIN_CMD="$TRAIN_CMD --force-config"
    fi
fi

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume /workspace/host/$RESUME_CHECKPOINT"
fi

if [ -n "$MIXED_PRECISION" ]; then
    TRAIN_CMD="$TRAIN_CMD --mixed-precision"
fi

echo -e "${BLUE}Starting BEVNeXt-SAM2 GPU training...${NC}"
echo -e "${YELLOW}GPU Memory before training:${NC}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i "$GPU_ID" | \
    awk -F', ' '{printf "  Used: %s MB / %s MB (%.1f%%)\n", $1, $2, ($1/$2)*100}'

echo ""
echo -e "${YELLOW}Docker command:${NC}"
echo -e "docker run $DOCKER_OPTS bevnext-sam2:latest $TRAIN_CMD"
echo ""

# Run training
echo -e "${GREEN}🚀 Launching GPU training...${NC}"
exec docker run $DOCKER_OPTS bevnext-sam2:latest $TRAIN_CMD