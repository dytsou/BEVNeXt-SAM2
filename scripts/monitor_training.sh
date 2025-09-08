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
        
        # Try to extract checkpoint info (requires Python)
        if command -v python3 &> /dev/null; then
            echo -e "  Info: $(python3 -c "
import torch, sys
try:
    ckpt = torch.load('outputs/training/checkpoint_latest.pth', map_location='cpu')
    epoch = ckpt.get('epoch', 'unknown')
    global_step = ckpt.get('global_step', ckpt.get('step', ckpt.get('batch_step', 'unknown')))
    timestamp = ckpt.get('timestamp', 'unknown')
    print(f'Epoch {epoch}, Step {global_step}, Saved: {timestamp[:19] if isinstance(timestamp, str) else timestamp}')
except:
    print('Could not read checkpoint info')
" 2>/dev/null || echo "Could not read checkpoint info")"
        fi
    fi
    
    if [ -f "outputs/training/checkpoint_best.pth" ]; then
        echo -e "${GREEN}✓ Best checkpoint found${NC}"
        echo -e "  Size: $(du -h outputs/training/checkpoint_best.pth | cut -f1)"
        
        # Try to extract best checkpoint info
        if command -v python3 &> /dev/null; then
            echo -e "  Info: $(python3 -c "
import torch, sys
try:
    ckpt = torch.load('outputs/training/checkpoint_best.pth', map_location='cpu')
    epoch = ckpt.get('epoch', 'unknown')
    val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', 'unknown'))
    print(f'Epoch {epoch}, Val Loss: {val_loss}')
except:
    print('Could not read checkpoint info')
" 2>/dev/null || echo "Could not read checkpoint info")"
        fi
    fi
    
    # Check for enhanced checkpoint metadata
    if [ -f "outputs/training/checkpoint_metadata.json" ]; then
        echo -e "${GREEN}✓ Checkpoint metadata found${NC}"
        if command -v python3 &> /dev/null; then
            echo -e "  Enhanced checkpoints: $(python3 -c "
import json
try:
    with open('outputs/training/checkpoint_metadata.json', 'r') as f:
        metadata = json.load(f)
    print(f'{len(metadata)} checkpoints tracked')
except:
    print('Could not read metadata')
" 2>/dev/null || echo "Could not read metadata")"
        fi
    fi
    
    # List all checkpoint files
    CHECKPOINT_COUNT=$(find outputs/training -name "checkpoint_*.pth*" 2>/dev/null | wc -l)
    if [ "$CHECKPOINT_COUNT" -gt 2 ]; then
        echo -e "${GREEN}✓ $CHECKPOINT_COUNT total checkpoint files found${NC}"
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
echo -e "  Resume training:      ./scripts/run.sh train --auto-resume"
echo -e "  Resume from specific: ./scripts/run.sh train --resume outputs/training/checkpoint_latest.pth"
echo -e "  Fresh training:       ./scripts/run.sh train --no-resume"