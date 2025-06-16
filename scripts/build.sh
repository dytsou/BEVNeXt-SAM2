#!/bin/bash

# BEVNeXt-SAM2 Docker Build Script
# This script builds the Docker image for BEVNeXt-SAM2

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="bevnext-sam2"
TAG="latest"
DOCKERFILE_PATH="Dockerfile"
BUILD_CONTEXT="."

# Print banner
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  BEVNeXt-SAM2 Docker Build Script    ${NC}"
echo -e "${BLUE}======================================${NC}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag|-t)
            TAG="$2"
            shift 2
            ;;
        --name|-n)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tag, -t TAG        Set image tag (default: latest)"
            echo "  --name, -n NAME      Set image name (default: bevnext-sam2)"
            echo "  --no-cache           Build without cache"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running or not accessible${NC}"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected. GPU support may not be available.${NC}"
fi

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo -e "${RED}Error: Dockerfile not found at $DOCKERFILE_PATH${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p data checkpoints outputs logs

# Build the Docker image
echo -e "${GREEN}Building Docker image: ${IMAGE_NAME}:${TAG}${NC}"
echo -e "${BLUE}Build context: ${BUILD_CONTEXT}${NC}"
echo -e "${BLUE}Dockerfile: ${DOCKERFILE_PATH}${NC}"

if [ ! -z "$NO_CACHE" ]; then
    echo -e "${YELLOW}Building without cache...${NC}"
fi

# Start build
DOCKER_BUILDKIT=1 docker build \
    $NO_CACHE \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "$DOCKERFILE_PATH" \
    "$BUILD_CONTEXT"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
    echo -e "${GREEN}Image: ${IMAGE_NAME}:${TAG}${NC}"
    
    # Show image size
    IMAGE_SIZE=$(docker images --format "table {{.Size}}" "${IMAGE_NAME}:${TAG}" | tail -n 1)
    echo -e "${BLUE}Image size: ${IMAGE_SIZE}${NC}"
    
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "${BLUE}1. Run the demo:${NC} docker-compose up bevnext-sam2"
    echo -e "${BLUE}2. Start development:${NC} docker-compose up dev"
    echo -e "${BLUE}3. Or use the run script:${NC} ./scripts/run.sh"
else
    echo -e "${RED}✗ Docker build failed!${NC}"
    exit 1
fi 