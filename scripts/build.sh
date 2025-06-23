#!/bin/bash

# BEVNeXt-SAM2 Docker Build Script (Fixed Version)
# This script builds the Docker image with proper dependency management

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="bevnext-sam2:latest"
BUILD_CONTEXT="."

# Print banner
echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  BEVNeXt-SAM2 Docker Build (Fixed)  ${NC}"
echo -e "${BLUE}=====================================${NC}"

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker daemon is not running${NC}"
        exit 1
    fi
    
    # Check NVIDIA Docker runtime (optional)
    if docker info 2>/dev/null | grep -q nvidia; then
        echo -e "${GREEN}✓ NVIDIA Docker runtime detected${NC}"
    else
        echo -e "${YELLOW}⚠ NVIDIA Docker runtime not detected - GPU support may not work${NC}"
    fi
    
    echo -e "${GREEN}✓ Prerequisites check passed${NC}"
}

# Function to clean up old images
cleanup_old_images() {
    echo -e "${BLUE}Cleaning up old images...${NC}"
    
    # Remove dangling images
    DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
    if [ ! -z "$DANGLING_IMAGES" ]; then
        echo -e "${YELLOW}Removing dangling images...${NC}"
        docker rmi $DANGLING_IMAGES || true
    fi
    
    # Remove old bevnext-sam2 images (keeping only the latest)
    OLD_IMAGES=$(docker images bevnext-sam2 -q | tail -n +2)
    if [ ! -z "$OLD_IMAGES" ]; then
        echo -e "${YELLOW}Removing old bevnext-sam2 images...${NC}"
        docker rmi $OLD_IMAGES || true
    fi
}

# Function to build the image
build_image() {
    echo -e "${BLUE}Building Docker image...${NC}"
    echo -e "${BLUE}This may take 10-20 minutes depending on your internet speed${NC}"
    
    # Build with BuildKit for better caching and performance
    DOCKER_BUILDKIT=1 docker build \
        --progress=plain \
        --tag $IMAGE_NAME \
        --file Dockerfile \
        $BUILD_CONTEXT
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Docker image built successfully!${NC}"
    else
        echo -e "${RED}✗ Docker build failed${NC}"
        exit 1
    fi
}

# Function to verify the build
verify_build() {
    echo -e "${BLUE}Verifying the build...${NC}"
    
    # Check if image exists
    if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Image exists${NC}"
    else
        echo -e "${RED}✗ Image not found${NC}"
        exit 1
    fi
    
    # Test basic imports
    echo -e "${BLUE}Testing Python imports...${NC}"
    docker run --rm $IMAGE_NAME python -c "
import torch
import mmcv
import mmdet
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MMCV: {mmcv.__version__}')
print(f'MMDet: {mmdet.__version__}')
print('All imports successful!')
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build verification passed${NC}"
    else
        echo -e "${RED}✗ Build verification failed${NC}"
        exit 1
    fi
}

# Main execution
main() {
    check_prerequisites
    cleanup_old_images
    build_image
    verify_build
    
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}  Build completed successfully!      ${NC}"
    echo -e "${GREEN}=====================================${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "1. Run the demo: ${GREEN}./scripts/run_fixed.sh demo${NC}"
    echo -e "2. Start development: ${GREEN}./scripts/run_fixed.sh dev${NC}"
    echo -e "3. Interactive shell: ${GREEN}./scripts/run_fixed.sh shell${NC}"
}

# Run main function
main 