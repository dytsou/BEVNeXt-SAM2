#!/bin/bash
# BEVNeXt-SAM2 Management Script
# Enhanced with container building, training, and validation capabilities

set -e

# Configuration
PROJECT_NAME="bevnext-sam2"
IMAGE_NAME="bevnext-sam2"
CONTAINER_NAME="bevnext-sam2-container"
DEFAULT_DATA_PATH="/data/nuscenes"  # Update this to your actual dataset path

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

# Docker options
DOCKER_OPTS="--rm -it"

print_help() {
    echo -e "${BLUE}BEVNeXt-SAM2 Management Script${NC}"
    echo ""
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Container Management:"
    echo "  build-container    - Build Docker container with optimized dataset handling"
    echo "  build-fast         - Build container without dataset copy (recommended)"
    echo "  build-full         - Build container with dataset copy (not recommended)"
    echo ""
    echo "Training Pipeline:"
    echo "  train              - Start training with volume-mounted dataset"
    echo "  train-download     - Start training with automatic dataset download"
    echo "  train-s3           - Start training with S3 dataset streaming"
    echo ""
    echo "Validation & Testing:"
    echo "  validate           - Validate trained model on test dataset"
    echo "  validate-nuscenes  - Comprehensive nuScenes dataset validation"
    echo "  test-model         - Run complete model testing suite"
    echo ""
    echo "Development:"
    echo "  dev                - Start development environment with Jupyter"
    echo "  demo               - Run demo with volume-mounted dataset"
    echo "  shell              - Start interactive shell"
    echo ""
    echo "Options:"
    echo "  --data-path PATH   - Path to nuScenes dataset (default: $DEFAULT_DATA_PATH)"
    echo "  --gpu              - Enable GPU support"
    echo "  --no-gpu           - Disable GPU support"
    echo "  --port PORT        - Expose additional port"
    echo "  --config CONFIG    - Training configuration file"
    echo "  --checkpoint PATH  - Path to model checkpoint for validation"
    echo "  --epochs N         - Number of training epochs"
    echo "  --batch-size N     - Training batch size"
    echo "  --dry-run          - Show commands without executing"
    echo ""
    echo "Examples:"
    echo "  $0 build-fast --gpu"
    echo "  $0 train --data-path /data/nuscenes --gpu --epochs 50"
    echo "  $0 validate --checkpoint outputs/checkpoints/latest.pth --gpu"
    echo "  $0 dev --data-path /data/nuscenes --gpu --jupyter"
}

remove_container() {
    local container_name="$1"
    if docker ps -a --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        echo -e "${YELLOW}Removing existing container: ${container_name}${NC}"
        docker rm -f "$container_name" >/dev/null 2>&1 || true
    fi
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

# Parse arguments
MODE=""
DATA_PATH=""
USE_GPU=true
DRY_RUN=false
TRAINING_CONFIG=""
CHECKPOINT_PATH=""
NUM_EPOCHS=""
BATCH_SIZE=""
EXTRA_PORTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        build-container|build-fast|build-full|train|train-download|train-s3|validate|validate-nuscenes|test-model|dev|demo|shell)
            MODE="$1"
            shift
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --no-gpu)
            USE_GPU=false
            shift
            ;;
        --port)
            EXTRA_PORTS="$EXTRA_PORTS -p $2:$2"
            shift 2
            ;;
        --config)
            TRAINING_CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
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
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo -e "${RED}Error: Mode is required${NC}"
    print_help
    exit 1
fi

# Set GPU options
if [[ "$USE_GPU" == true ]]; then
    setup_gpu
fi

# Set data path
if [[ -z "$DATA_PATH" ]]; then
    DATA_PATH="$DEFAULT_DATA_PATH"
fi

# Convert data path to absolute path for Docker volume mounting
if [[ -d "$DATA_PATH" ]]; then
    ABS_DATA_PATH="$(realpath "$DATA_PATH")"
    echo -e "${GREEN}Using dataset: $ABS_DATA_PATH${NC}"
else
    ABS_DATA_PATH="$DATA_PATH"
    echo -e "${YELLOW}Dataset path not found locally: $DATA_PATH${NC}"
    echo -e "${YELLOW}Will attempt to use as-is (may be a volume or remote path)${NC}"
fi

# Create necessary directories
mkdir -p "$ABS_OUTPUTS_PATH" "$ABS_LOGS_PATH"

# Build base docker command
build_docker_cmd() {
    local cmd="docker run $DOCKER_OPTS"
    cmd="$cmd --name $CONTAINER_NAME"
    cmd="$cmd -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2"
    cmd="$cmd -v $ABS_OUTPUTS_PATH:/workspace/outputs"
    cmd="$cmd -v $ABS_LOGS_PATH:/workspace/logs"
    cmd="$cmd -w /workspace/bevnext-sam2"
    
    # Add data volume mount
    if [[ "$MODE" == *"train"* ]] || [[ "$MODE" == "dev" ]] || [[ "$MODE" == "demo" ]] || [[ "$MODE" == "validate" ]]; then
        cmd="$cmd -v $ABS_DATA_PATH:/workspace/data/nuscenes:ro"
    fi
    
    # Add extra ports
    if [[ -n "$EXTRA_PORTS" ]]; then
        cmd="$cmd $EXTRA_PORTS"
    fi
    
    # Add environment variables
    cmd="$cmd -e NUSCENES_DATA_ROOT=/workspace/data/nuscenes"
    cmd="$cmd -e PYTHONPATH=/workspace/bevnext-sam2"
    
    echo "$cmd"
}

# Container building modes
case $MODE in
    build-container|build-fast)
        echo -e "${GREEN}Building optimized Docker container (without dataset copy)...${NC}"
        echo -e "${BLUE}This is the recommended approach - dataset will be mounted at runtime${NC}"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "docker build -t $IMAGE_NAME ."
        else
            docker build -t "$IMAGE_NAME" .
            echo -e "${GREEN}Container built successfully!${NC}"
            echo -e "${CYAN}Next steps:${NC}"
            echo -e "  • Run training: $0 train --data-path $DATA_PATH --gpu"
            echo -e "  • Start development: $0 dev --data-path $DATA_PATH --gpu"
            echo -e "  • Run validation: $0 validate --checkpoint outputs/checkpoints/latest.pth --gpu"
        fi
        ;;
        
    build-full)
        echo -e "${YELLOW}Building full Docker container (with dataset copy)...${NC}"
        echo -e "${RED}Warning: This will create a very large image and take a long time${NC}"
        echo -e "${YELLOW}Consider using build-fast instead${NC}"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "docker build -t $IMAGE_NAME-full --build-arg COPY_DATASET=true ."
        else
            docker build -t "$IMAGE_NAME-full" --build-arg COPY_DATASET=true .
            echo -e "${GREEN}Full container built successfully!${NC}"
        fi
        ;;
        
    train)
        echo -e "${GREEN}Starting BEVNeXt-SAM2 training with nuScenes dataset...${NC}"
        echo -e "${BLUE}Dataset path: $DATA_PATH${NC}"
        
        if [[ ! -d "$DATA_PATH" ]]; then
            echo -e "${RED}Error: Dataset path does not exist: $DATA_PATH${NC}"
            echo -e "${YELLOW}Please ensure the nuScenes dataset is available at this location${NC}"
            exit 1
        fi
        
        remove_container "$CONTAINER_NAME"
        
        # Build training command
        TRAIN_CMD="python training/train_bevnext_sam2_nuscenes.py --data-root /workspace/data/nuscenes"
        
        if [[ -n "$TRAINING_CONFIG" ]]; then
            TRAIN_CMD="$TRAIN_CMD --config $TRAINING_CONFIG"
        fi
        
        if [[ -n "$NUM_EPOCHS" ]]; then
            TRAIN_CMD="$TRAIN_CMD --epochs $NUM_EPOCHS"
        fi
        
        if [[ -n "$BATCH_SIZE" ]]; then
            TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
        fi
        
        TRAIN_CMD="$TRAIN_CMD --mixed-precision"
        
        DOCKER_CMD=$(build_docker_cmd)
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "$DOCKER_CMD bash -c \"$TRAIN_CMD\""
        else
            echo -e "${GREEN}🚀 Starting training...${NC}"
            eval "$DOCKER_CMD bash -c \"$TRAIN_CMD\""
        fi
        ;;
        
    train-download)
        echo -e "${GREEN}Starting training with automatic dataset download...${NC}"
        
        remove_container "$CONTAINER_NAME"
        
        DOCKER_CMD=$(build_docker_cmd)
        DOCKER_CMD="$DOCKER_CMD -e DOWNLOAD_NUSCENES=true"
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        TRAIN_CMD="python training/train_bevnext_sam2_nuscenes.py --data-root /workspace/data/nuscenes --mixed-precision"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "$DOCKER_CMD bash -c \"$TRAIN_CMD\""
        else
            echo -e "${GREEN}🚀 Starting training with dataset download...${NC}"
            eval "$DOCKER_CMD bash -c \"$TRAIN_CMD\""
        fi
        ;;
        
    validate)
        echo -e "${GREEN}Validating trained model...${NC}"
        
        if [[ -z "$CHECKPOINT_PATH" ]]; then
            echo -e "${YELLOW}No checkpoint specified, using latest checkpoint${NC}"
            CHECKPOINT_PATH="outputs/checkpoints/latest.pth"
        fi
        
        remove_container "$CONTAINER_NAME"
        
        DOCKER_CMD=$(build_docker_cmd)
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        VALIDATE_CMD="python validation/validate_model.py --checkpoint $CHECKPOINT_PATH --data-root /workspace/data/nuscenes"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "$DOCKER_CMD bash -c \"$VALIDATE_CMD\""
        else
            echo -e "${GREEN}🔍 Starting model validation...${NC}"
            eval "$DOCKER_CMD bash -c \"$VALIDATE_CMD\""
        fi
        ;;
        
    validate-nuscenes)
        echo -e "${GREEN}Validating nuScenes dataset integrity...${NC}"
        
        remove_container "$CONTAINER_NAME"
        
        DOCKER_CMD=$(build_docker_cmd)
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "$DOCKER_CMD bash -c \"python validation/nuscenes_validator.py --data-root /workspace/data/nuscenes --version v1.0-trainval --output-dir /workspace/outputs/validation_reports\""
        else
            echo -e "${GREEN}🔍 Starting nuScenes validation...${NC}"
            eval "$DOCKER_CMD bash -c \"python validation/nuscenes_validator.py --data-root /workspace/data/nuscenes --version v1.0-trainval --output-dir /workspace/outputs/validation_reports\""
        fi
        ;;
        
    test-model)
        echo -e "${GREEN}Running complete model testing suite...${NC}"
        
        remove_container "$CONTAINER_NAME"
        
        DOCKER_CMD=$(build_docker_cmd)
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "$DOCKER_CMD bash -c \"python test_training_setup.py\""
        else
            echo -e "${GREEN}🧪 Running model tests...${NC}"
            eval "$DOCKER_CMD bash -c \"python test_training_setup.py\""
        fi
        ;;
        
    dev)
        echo -e "${GREEN}Starting development environment with Jupyter...${NC}"
        
        remove_container "$CONTAINER_NAME"
        
        DOCKER_CMD=$(build_docker_cmd)
        DOCKER_CMD="$DOCKER_CMD -p 8888:8888 -p 6006:6006"
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "$DOCKER_CMD bash -c \"jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\""
        else
            echo -e "${GREEN}🚀 Starting Jupyter development environment...${NC}"
            echo -e "${CYAN}Jupyter will be available at: http://localhost:8888${NC}"
            eval "$DOCKER_CMD bash -c \"jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\""
        fi
        ;;
        
    demo)
        echo -e "${GREEN}Running BEVNeXt-SAM2 demo...${NC}"
        
        remove_container "$CONTAINER_NAME"
        
        DOCKER_CMD=$(build_docker_cmd)
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "$DOCKER_CMD bash -c \"python examples/demo_fusion.py\""
        else
            echo -e "${GREEN}🎬 Starting demo...${NC}"
            eval "$DOCKER_CMD bash -c \"python examples/demo_fusion.py\""
        fi
        ;;
        
    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        
        remove_container "$CONTAINER_NAME"
        
        DOCKER_CMD=$(build_docker_cmd)
        DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "$DOCKER_CMD bash"
        else
            echo -e "${GREEN}🐚 Starting interactive shell...${NC}"
            eval "$DOCKER_CMD bash"
        fi
        ;;
        
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        print_help
        exit 1
        ;;
esac