#!/bin/bash
# Docker Volume Mount Initialization Script for BEVNeXt-SAM2
# Ensures proper directory structure and permissions for checkpoint persistence

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}🚀 Initializing Docker Volume Mounts for BEVNeXt-SAM2${NC}"
echo "=================================================="

# Define directory structure
OUTPUTS_DIR="$PROJECT_ROOT/outputs"
CHECKPOINTS_DIR="$OUTPUTS_DIR/checkpoints"
TRAINING_DIR="$OUTPUTS_DIR/training"
EVALUATION_DIR="$OUTPUTS_DIR/evaluation"
TENSORBOARD_DIR="$OUTPUTS_DIR/tensorboard"
VALIDATION_DIR="$OUTPUTS_DIR/validation_reports"
LOGS_DIR="$PROJECT_ROOT/logs"
LEGACY_CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"

echo -e "${YELLOW}📁 Creating directory structure...${NC}"

# Create all required directories
mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$TRAINING_DIR" 
mkdir -p "$EVALUATION_DIR"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$VALIDATION_DIR"
mkdir -p "$LOGS_DIR"

# Create legacy checkpoint directory for backward compatibility
mkdir -p "$LEGACY_CHECKPOINTS_DIR"

echo -e "${GREEN}✅ Created directories:${NC}"
echo "   📦 $CHECKPOINTS_DIR/"
echo "   📊 $TRAINING_DIR/"
echo "   📈 $EVALUATION_DIR/" 
echo "   📉 $TENSORBOARD_DIR/"
echo "   📋 $VALIDATION_DIR/"
echo "   📜 $LOGS_DIR/"
echo "   🔗 $LEGACY_CHECKPOINTS_DIR/"

echo ""
echo -e "${YELLOW}🔐 Setting permissions for Docker access...${NC}"

# Set proper permissions for Docker container access
# Using 755 to allow read/write/execute for owner, read/execute for group/others
chmod -R 755 "$OUTPUTS_DIR" "$LOGS_DIR" "$LEGACY_CHECKPOINTS_DIR" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Could not set permissions (non-critical)${NC}"
}

echo -e "${GREEN}✅ Permissions set${NC}"

echo ""
echo -e "${YELLOW}🐳 Docker volume mount mappings:${NC}"
echo "   Host: $OUTPUTS_DIR/ → Container: /workspace/outputs/"
echo "   Host: $CHECKPOINTS_DIR/ → Container: /workspace/outputs/checkpoints/"
echo "   Host: $TRAINING_DIR/ → Container: /workspace/outputs/training/"
echo "   Host: $EVALUATION_DIR/ → Container: /workspace/outputs/evaluation/"
echo "   Host: $LOGS_DIR/ → Container: /workspace/logs/"

echo ""
echo -e "${YELLOW}📊 Current directory status:${NC}"
ls -la "$OUTPUTS_DIR/" 2>/dev/null || echo "   Outputs directory just created"
echo ""
echo -e "${BLUE}💾 Disk usage:${NC}"
du -sh "$OUTPUTS_DIR" "$LOGS_DIR" 2>/dev/null || echo "   Directories just created (0 bytes)"

echo ""
echo -e "${GREEN}🎯 Ready for Docker training!${NC}"
echo ""
echo "Next steps:"
echo "  1. Start training: make train-ddp"
echo "  2. Monitor progress: make monitor"
echo "  3. Check status: make status"
echo ""
echo "Your checkpoints will be saved to:"
echo "  🎯 $CHECKPOINTS_DIR/"
echo "  📊 Training logs: $TRAINING_DIR/"
echo ""
echo -e "${BLUE}=================================================${NC}"
