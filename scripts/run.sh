#!/bin/bash

# BEVNeXt-SAM2 Docker Run Script
# This script provides easy access to different ways of running the container

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="bevnext-sam2:latest"
CONTAINER_NAME="bevnext-sam2-interactive"

# Print banner
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  BEVNeXt-SAM2 Docker Run Script     ${NC}"
echo -e "${BLUE}=====================================${NC}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  demo                 Run the demo script"
    echo "  dev                  Start development environment with Jupyter"
    echo "  shell                Start interactive shell in container"
    echo "  train                Start training environment"
    echo "  inference            Start inference environment"
    echo "  tensorboard          Start TensorBoard server"
    echo ""
    echo "Options:"
    echo "  --gpu GPU_ID         Specify GPU ID (default: 0)"
    echo "  --port PORT          Jupyter port (default: 8888)"
    echo "  --data PATH          Path to data directory"
    echo "  --checkpoints PATH   Path to checkpoints directory"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 demo                    # Run demo"
    echo "  $0 dev --port 8889         # Start Jupyter on port 8889"
    echo "  $0 shell                   # Interactive shell"
    echo "  $0 train --gpu 1           # Training on GPU 1"
}

# Parse command line arguments
MODE=""
GPU_ID="0"
JUPYTER_PORT="8888"
DATA_PATH="./data"
CHECKPOINTS_PATH="./checkpoints"

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

MODE=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --port)
            JUPYTER_PORT="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --checkpoints)
            CHECKPOINTS_PATH="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running or not accessible${NC}"
    exit 1
fi

# Check if image exists
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker image $IMAGE_NAME not found${NC}"
    echo -e "${YELLOW}Please build the image first: ./scripts/build.sh${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p "$DATA_PATH" "$CHECKPOINTS_PATH" "./outputs" "./logs"

# Common Docker run options
DOCKER_OPTS="--rm -it"
DOCKER_OPTS="$DOCKER_OPTS --runtime=nvidia"
DOCKER_OPTS="$DOCKER_OPTS -e NVIDIA_VISIBLE_DEVICES=all"
DOCKER_OPTS="$DOCKER_OPTS -e CUDA_VISIBLE_DEVICES=$GPU_ID"
DOCKER_OPTS="$DOCKER_OPTS -v $(realpath $DATA_PATH):/workspace/data"
DOCKER_OPTS="$DOCKER_OPTS -v $(realpath $CHECKPOINTS_PATH):/workspace/checkpoints"
DOCKER_OPTS="$DOCKER_OPTS -v $(realpath ./outputs):/workspace/outputs"
DOCKER_OPTS="$DOCKER_OPTS -v $(realpath ./logs):/workspace/logs"

# Run based on mode
case $MODE in
    demo)
        echo -e "${GREEN}Running BEVNeXt-SAM2 demo...${NC}"
        docker run $DOCKER_OPTS \
            --name "${CONTAINER_NAME}-demo" \
            $IMAGE_NAME \
            python examples/demo_fusion.py
        ;;
        
    dev)
        echo -e "${GREEN}Starting development environment with Jupyter Lab...${NC}"
        echo -e "${BLUE}Jupyter will be available at: http://localhost:${JUPYTER_PORT}${NC}"
        docker run $DOCKER_OPTS \
            --name "${CONTAINER_NAME}-dev" \
            -p ${JUPYTER_PORT}:8888 \
            -p 6006:6006 \
            -v $(pwd):/workspace/bevnext-sam2 \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                       --NotebookApp.token='' --NotebookApp.password=''
        ;;
        
    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        docker run $DOCKER_OPTS \
            --name "${CONTAINER_NAME}-shell" \
            -v $(pwd):/workspace/bevnext-sam2 \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash
        ;;
        
    train)
        echo -e "${GREEN}Starting training environment...${NC}"
        echo -e "${BLUE}Use this environment for training models${NC}"
        docker run $DOCKER_OPTS \
            --name "${CONTAINER_NAME}-train" \
            -v $(pwd)/configs:/workspace/bevnext-sam2/configs \
            $IMAGE_NAME \
            bash
        ;;
        
    inference)
        echo -e "${GREEN}Starting inference environment...${NC}"
        echo -e "${BLUE}Use this environment for running inference${NC}"
        docker run $DOCKER_OPTS \
            --name "${CONTAINER_NAME}-inference" \
            -v $(pwd)/configs:/workspace/bevnext-sam2/configs \
            $IMAGE_NAME \
            bash
        ;;
        
    tensorboard)
        echo -e "${GREEN}Starting TensorBoard server...${NC}"
        echo -e "${BLUE}TensorBoard will be available at: http://localhost:6006${NC}"
        docker run $DOCKER_OPTS \
            --name "${CONTAINER_NAME}-tensorboard" \
            -p 6006:6006 \
            $IMAGE_NAME \
            tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006
        ;;
        
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        show_usage
        exit 1
        ;;
esac

echo -e "${GREEN}✓ Container started successfully!${NC}" 