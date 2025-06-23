#!/bin/bash

# BEVNeXt-SAM2 Training Monitor Script
# This script monitors the training progress

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  BEVNeXt-SAM2 Training Monitor       ${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if training container is running
echo -e "${BLUE}Checking training status...${NC}"
CONTAINER_COUNT=$(docker ps --filter ancestor=bevnext-sam2:latest --format "table {{.ID}}\t{{.Status}}" | grep -v CONTAINER | wc -l)

if [ "$CONTAINER_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Training container is running!${NC}"
    
    # Show container info
    echo -e "\n${YELLOW}Container Status:${NC}"
    docker ps --filter ancestor=bevnext-sam2:latest --format "table {{.ID}}\t{{.Status}}\t{{.Names}}"
    
    # Show recent logs
    echo -e "\n${YELLOW}Recent Training Logs:${NC}"
    CONTAINER_ID=$(docker ps --filter ancestor=bevnext-sam2:latest --format "{{.ID}}" | head -1)
    docker logs --tail 20 "$CONTAINER_ID"
    
else
    echo -e "${RED}❌ No training container found${NC}"
    echo -e "Training may have completed or failed."
fi

# Check for output files
echo -e "\n${BLUE}Checking output files...${NC}"

if [ -d "outputs/training" ]; then
    echo -e "${GREEN}✓ Training output directory exists${NC}"
    
    # Check for tensorboard logs
    if [ -d "outputs/training/tensorboard" ]; then
        echo -e "${GREEN}✓ TensorBoard logs found${NC}"
        echo -e "  View logs with: tensorboard --logdir outputs/training/tensorboard"
    fi
    
    # Check for checkpoints
    if [ -f "outputs/training/checkpoint_latest.pth" ]; then
        echo -e "${GREEN}✓ Latest checkpoint found${NC}"
        echo -e "  Size: $(du -h outputs/training/checkpoint_latest.pth | cut -f1)"
    fi
    
    if [ -f "outputs/training/checkpoint_best.pth" ]; then
        echo -e "${GREEN}✓ Best checkpoint found${NC}"
        echo -e "  Size: $(du -h outputs/training/checkpoint_best.pth | cut -f1)"
    fi
    
    # Check for training log
    if [ -f "outputs/training/training.log" ]; then
        echo -e "${GREEN}✓ Training log found${NC}"
        echo -e "\n${YELLOW}Latest log entries:${NC}"
        tail -10 outputs/training/training.log
    fi
    
    # Show directory contents
    echo -e "\n${YELLOW}Training directory contents:${NC}"
    ls -la outputs/training/
    
else
    echo -e "${YELLOW}⚠ Training output directory not found${NC}"
fi

# Check GPU usage if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n${BLUE}GPU Usage:${NC}"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
fi

echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}Monitoring Complete${NC}"
echo -e "${BLUE}======================================${NC}"

# Provide helpful commands
echo -e "\n${YELLOW}Useful Commands:${NC}"
echo -e "  Monitor live logs:    docker logs -f \$(docker ps --filter ancestor=bevnext-sam2:latest --format \"{{.ID}}\" | head -1)"
echo -e "  Stop training:        docker stop \$(docker ps --filter ancestor=bevnext-sam2:latest --format \"{{.ID}}\" | head -1)"
echo -e "  View tensorboard:     tensorboard --logdir outputs/training/tensorboard"
echo -e "  Resume training:      ./scripts/train.sh --resume outputs/training/checkpoint_latest.pth"