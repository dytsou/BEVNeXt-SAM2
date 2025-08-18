#!/bin/bash
# Quick Model Validation Script for BEVNeXt-SAM2
# 
# This script provides a convenient way to quickly validate trained models
# with common validation tasks and sensible defaults.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CHECKPOINT_PATH="checkpoints/latest.pth"
DATA_ROOT="data/nuscenes"
OUTPUT_DIR="outputs/quick_validation"
MAX_SAMPLES=100
DEVICE="auto"
USE_DOCKER=false
GENERATE_VIZ=false
RUN_NUSCENES_EVAL=false
VERBOSE=false

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_usage() {
    echo -e "${BLUE}BEVNeXt-SAM2 Quick Validation Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --checkpoint PATH    Path to model checkpoint (default: $CHECKPOINT_PATH)"
    echo "  -d, --data-root PATH     Path to nuScenes dataset (default: $DATA_ROOT)"
    echo "  -o, --output-dir PATH    Output directory (default: $OUTPUT_DIR)"
    echo "  -s, --max-samples N      Maximum samples to validate (default: $MAX_SAMPLES)"
    echo "  --device DEVICE          Device to use: auto, cuda, cpu (default: $DEVICE)"
    echo "  --docker                 Use Docker for validation"
    echo "  --full                   Run full validation with nuScenes eval and visualizations"
    echo "  --viz                    Generate visualizations"
    echo "  --nuscenes-eval          Run official nuScenes evaluation"
    echo "  -v, --verbose            Verbose output"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Quick validation with defaults"
    echo "  $0 -c models/best.pth -d /data/nuscenes      # Custom checkpoint and data path"
    echo "  $0 --full --docker                           # Full validation in Docker"
    echo "  $0 -s 50 --viz                              # Quick validation with visualizations"
}

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        -d|--data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --full)
            RUN_NUSCENES_EVAL=true
            GENERATE_VIZ=true
            MAX_SAMPLES=500
            shift
            ;;
        --viz)
            GENERATE_VIZ=true
            shift
            ;;
        --nuscenes-eval)
            RUN_NUSCENES_EVAL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate inputs
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    error "Checkpoint file not found: $CHECKPOINT_PATH"
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    error "Data directory not found: $DATA_ROOT"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

log "Starting BEVNeXt-SAM2 model validation..."
log "Configuration:"
log "  └─ Checkpoint: $CHECKPOINT_PATH"
log "  └─ Data root: $DATA_ROOT"
log "  └─ Output dir: $OUTPUT_DIR"
log "  └─ Max samples: $MAX_SAMPLES"
log "  └─ Device: $DEVICE"
log "  └─ Use Docker: $USE_DOCKER"
log "  └─ Generate viz: $GENERATE_VIZ"
log "  └─ nuScenes eval: $RUN_NUSCENES_EVAL"

# Build validation command
if [[ "$USE_DOCKER" == true ]]; then
    log "Running validation in Docker..."
    
    # Check if Docker image exists
    if ! docker image inspect bevnext-sam2 >/dev/null 2>&1; then
        warn "Docker image 'bevnext-sam2' not found. Building..."
        cd "$PROJECT_ROOT"
        ./scripts/run.sh build-fast --gpu
    fi
    
    # Build Docker command
    DOCKER_CMD="docker run --rm"
    
    # Add GPU support if available and requested
    if [[ "$DEVICE" == "auto" || "$DEVICE" == "cuda" ]]; then
        if command -v nvidia-docker >/dev/null 2>&1 || docker info | grep -q nvidia; then
            DOCKER_CMD="$DOCKER_CMD --gpus all"
            log "GPU support enabled"
        else
            warn "GPU requested but not available, using CPU"
            DEVICE="cpu"
        fi
    fi
    
    # Mount volumes
    DOCKER_CMD="$DOCKER_CMD -v $PROJECT_ROOT:/workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD -v $(realpath $DATA_ROOT):/workspace/data/nuscenes:ro"
    DOCKER_CMD="$DOCKER_CMD -v $(realpath $OUTPUT_DIR):/workspace/outputs"
    DOCKER_CMD="$DOCKER_CMD -w /workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD -e PYTHONPATH=/workspace/bevnext-sam2"
    DOCKER_CMD="$DOCKER_CMD bevnext-sam2"
    
    # Build validation command
    VAL_CMD="python validation/validate_model.py"
    VAL_CMD="$VAL_CMD --checkpoint $(realpath $CHECKPOINT_PATH)"
    VAL_CMD="$VAL_CMD --data-root /workspace/data/nuscenes"
    VAL_CMD="$VAL_CMD --output-dir /workspace/outputs"
    VAL_CMD="$VAL_CMD --max-samples $MAX_SAMPLES"
    
    if [[ "$RUN_NUSCENES_EVAL" == true ]]; then
        VAL_CMD="$VAL_CMD --run-nuscenes-eval"
    fi
    
    if [[ "$GENERATE_VIZ" == true ]]; then
        VAL_CMD="$VAL_CMD --generate-viz"
    fi
    
    # Execute Docker command
    FULL_CMD="$DOCKER_CMD $VAL_CMD"
    
else
    log "Running validation natively..."
    
    cd "$PROJECT_ROOT"
    
    # Build validation command
    VAL_CMD="python validation/validate_model.py"
    VAL_CMD="$VAL_CMD --checkpoint $CHECKPOINT_PATH"
    VAL_CMD="$VAL_CMD --data-root $DATA_ROOT"
    VAL_CMD="$VAL_CMD --output-dir $OUTPUT_DIR"
    VAL_CMD="$VAL_CMD --max-samples $MAX_SAMPLES"
    
    if [[ "$RUN_NUSCENES_EVAL" == true ]]; then
        VAL_CMD="$VAL_CMD --run-nuscenes-eval"
    fi
    
    if [[ "$GENERATE_VIZ" == true ]]; then
        VAL_CMD="$VAL_CMD --generate-viz"
    fi
    
    FULL_CMD="$VAL_CMD"
fi

# Show command if verbose
if [[ "$VERBOSE" == true ]]; then
    log "Running command: $FULL_CMD"
fi

# Execute validation
log "Starting model validation..."
eval "$FULL_CMD"

if [[ $? -eq 0 ]]; then
    log "✅ Model validation completed successfully!"
    log "Results saved to: $OUTPUT_DIR"
    
    # Show key results if available
    METRICS_FILE="$OUTPUT_DIR/validation_metrics.json"
    if [[ -f "$METRICS_FILE" ]]; then
        log "Key validation metrics:"
        
        # Extract and display key metrics using Python
        python3 -c "
import json
import sys
try:
    with open('$METRICS_FILE', 'r') as f:
        metrics = json.load(f)
    
    key_metrics = ['mean_iou', 'classification_accuracy', 'max_iou']
    for metric in key_metrics:
        if metric in metrics:
            print(f'  └─ {metric}: {metrics[metric]:.4f}')
except Exception as e:
    print(f'  └─ Could not parse metrics: {e}')
" 2>/dev/null || echo "  └─ Metrics file found but could not parse"
    fi
    
    # Show additional output files
    if [[ -d "$OUTPUT_DIR" ]]; then
        log "Generated files:"
        find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.png" -o -name "*.html" | head -10 | while read file; do
            echo "  └─ $(basename "$file")"
        done
    fi
    
else
    error "❌ Model validation failed! Check the logs above for details."
fi

log "Validation script completed."
