#!/bin/bash

# BEVNeXt-SAM2 Docker Validation Script
# This script validates the Docker setup and runs training/testing

set -e

echo "🚀 BEVNeXt-SAM2 Docker Validation"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_info() {
    echo -e "${BLUE}🔍${NC} $1"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not available. Please install Docker first."
    exit 1
fi

print_status "Docker is available"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    print_warning "docker-compose not found. Using docker compose instead."
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
    print_status "docker-compose is available"
fi

# Step 1: Build the Docker image
print_info "Step 1: Building Docker image..."
if docker build -t bevnext-sam2:latest .; then
    print_status "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Step 2: Test basic container startup
print_info "Step 2: Testing container startup..."
if docker run --rm bevnext-sam2:latest python -c "print('Container startup successful')"; then
    print_status "Container startup test passed"
else
    print_error "Container startup test failed"
    exit 1
fi

# Step 3: Test Python imports inside container
print_info "Step 3: Testing Python imports..."
if docker run --rm bevnext-sam2:latest python3 test_training_setup.py; then
    print_status "Python imports test passed"
else
    print_warning "Python imports test had issues (check output above)"
fi

# Step 4: Test training script
print_info "Step 4: Testing training script..."
if docker run --rm bevnext-sam2:latest python3 training/train_bevnext_sam2.py --help; then
    print_status "Training script is accessible"
else
    print_warning "Training script test had issues"
fi

# Step 5: Test demo script
print_info "Step 5: Testing demo script..."
if docker run --rm bevnext-sam2:latest python3 examples/demo_fusion.py --help; then
    print_status "Demo script is accessible"
else
    print_warning "Demo script test had issues"
fi

# Step 6: Test with docker-compose
print_info "Step 6: Testing docker-compose setup..."
if $COMPOSE_CMD config > /dev/null 2>&1; then
    print_status "docker-compose configuration is valid"
else
    print_error "docker-compose configuration is invalid"
    exit 1
fi

# Step 7: Test GPU support (if available)
print_info "Step 7: Testing GPU support..."
if docker run --rm --gpus all bevnext-sam2:latest python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    print_status "GPU support test passed"
else
    print_warning "GPU support not available (this is OK if you don't have GPU)"
fi

echo ""
echo "🎉 Docker validation completed!"
echo ""
echo "Next steps:"
echo "1. To run training: ./scripts/run.sh train"
echo "2. To run demo: ./scripts/run.sh demo"
echo "3. To start development: ./scripts/run.sh dev"
echo "4. To run with docker-compose: $COMPOSE_CMD up bevnext-sam2"
echo ""
echo "For more information, see README.md and DOCKER.md"