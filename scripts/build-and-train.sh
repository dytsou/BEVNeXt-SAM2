#!/bin/bash
# BEVNeXt-SAM2 Complete Pipeline Script
# Builds container and runs training + validation pipeline

set -e

# Configuration
PROJECT_NAME="bevnext-sam2"
IMAGE_NAME="bevnext-sam2"
CONTAINER_NAME="bevnext-sam2-pipeline"
DEFAULT_DATA_PATH="/data/nuscenes"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_help() {
    echo -e "${BLUE}BEVNeXt-SAM2 Complete Pipeline Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script performs the complete pipeline:"
    echo "  1. Build optimized Docker container"
    echo "  2. Validate nuScenes dataset"
    echo "  3. Train BEVNeXt-SAM2 model"
    echo "  4. Validate trained model"
    echo "  5. Generate evaluation reports"
    echo ""
    echo "Options:"
    echo "  --data-path PATH    - Path to nuScenes dataset (default: $DEFAULT_DATA_PATH)"
    echo "  --gpu               - Enable GPU support"
    echo "  --no-gpu            - Disable GPU support"
    echo "  --epochs N          - Number of training epochs (default: 50)"
    echo "  --batch-size N      - Training batch size (default: 1)"
    echo "  --config FILE       - Training configuration file"
    echo "  --output-dir DIR    - Output directory (default: outputs/pipeline)"
    echo "  --skip-build        - Skip container building"
    echo "  --skip-validation   - Skip dataset validation"
    echo "  --skip-training     - Skip model training"
    echo "  --skip-eval         - Skip model evaluation"
    echo "  --dry-run           - Show commands without executing"
    echo "  --help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --data-path /data/nuscenes --gpu --epochs 100"
    echo "  $0 --data-path /data/nuscenes --gpu --config configs/training_config.json"
    echo "  $0 --skip-build --data-path /data/nuscenes --gpu"
}

# Parse arguments
DATA_PATH="$DEFAULT_DATA_PATH"
USE_GPU=true
NUM_EPOCHS=50
BATCH_SIZE=1
TRAINING_CONFIG=""
OUTPUT_DIR="outputs/pipeline"
SKIP_BUILD=false
SKIP_VALIDATION=false
SKIP_TRAINING=false
SKIP_EVAL=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --config)
            TRAINING_CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
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

# Set data path
if [[ -z "$DATA_PATH" ]]; then
    DATA_PATH="$DEFAULT_DATA_PATH"
fi

# Convert data path to absolute path for Docker volume mounting
if [[ -d "$DATA_PATH" ]]; then
    ABS_DATA_PATH="$(realpath "$DATA_PATH")"
    log_step "Using dataset: $ABS_DATA_PATH"
else
    ABS_DATA_PATH="$DATA_PATH"
    log_warning "Dataset path not found locally: $DATA_PATH"
    log_warning "Will attempt to use as-is (may be a volume or remote path)"
fi

# Validate inputs
if [[ ! -d "$ABS_DATA_PATH" ]]; then
    echo -e "${RED}Error: Dataset path does not exist: $ABS_DATA_PATH${NC}"
    echo -e "${YELLOW}Please ensure the nuScenes dataset is available at this location${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log function
log_step() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check GPU availability
check_gpu() {
    if [[ "$USE_GPU" == true ]]; then
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
            log_step "GPU detected - enabling GPU support"
            return 0
        else
            log_warning "GPU requested but not available - falling back to CPU"
            USE_GPU=false
            return 1
        fi
    fi
    return 0
}

# Build Docker container
build_container() {
    if [[ "$SKIP_BUILD" == true ]]; then
        log_step "Skipping container build (--skip-build specified)"
        return 0
    fi
    
    log_step "Building optimized Docker container..."
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "docker build -t $IMAGE_NAME ."
    else
        docker build -t "$IMAGE_NAME" .
        if [[ $? -eq 0 ]]; then
            log_step "Container built successfully!"
        else
            log_error "Container build failed!"
            exit 1
        fi
    fi
}

# Validate dataset
validate_dataset() {
    if [[ "$SKIP_VALIDATION" == true ]]; then
        log_step "Skipping dataset validation (--skip-validation specified)"
        return 0
    fi
    
    log_step "Validating nuScenes dataset..."
    
    # Build validation command
    VALIDATE_CMD="docker run --rm"
    if [[ "$USE_GPU" == true ]]; then
        VALIDATE_CMD="$VALIDATE_CMD --gpus all"
    fi
    VALIDATE_CMD="$VALIDATE_CMD -v $PROJECT_ROOT:/workspace/bevnext-sam2"
    VALIDATE_CMD="$VALIDATE_CMD -v $ABS_DATA_PATH:/workspace/data/nuscenes:ro"
    VALIDATE_CMD="$VALIDATE_CMD -v $OUTPUT_DIR:/workspace/outputs"
    VALIDATE_CMD="$VALIDATE_CMD -w /workspace/bevnext-sam2"
    VALIDATE_CMD="$VALIDATE_CMD -e NUSCENES_DATA_ROOT=/workspace/data/nuscenes"
    VALIDATE_CMD="$VALIDATE_CMD -e PYTHONPATH=/workspace/bevnext-sam2"
    VALIDATE_CMD="$VALIDATE_CMD $IMAGE_NAME"
    VALIDATE_CMD="$VALIDATE_CMD python validation/nuscenes_validator.py --data-root /workspace/data/nuscenes --version v1.0-trainval --output-dir /workspace/outputs/validation_reports"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "$VALIDATE_CMD"
    else
        eval "$VALIDATE_CMD"
        if [[ $? -eq 0 ]]; then
            log_step "Dataset validation completed!"
        else
            log_error "Dataset validation failed!"
            exit 1
        fi
    fi
}

# Train model
train_model() {
    if [[ "$SKIP_TRAINING" == true ]]; then
        log_step "Skipping model training (--skip-training specified)"
        return 0
    fi
    
    log_step "Starting BEVNeXt-SAM2 training..."
    log_step "Training configuration:"
    log_step "  - Epochs: $NUM_EPOCHS"
    log_step "  - Batch size: $BATCH_SIZE"
    log_step "  - GPU: $USE_GPU"
    if [[ -n "$TRAINING_CONFIG" ]]; then
        log_step "  - Config file: $TRAINING_CONFIG"
    fi
    
    # Build training command
    TRAIN_CMD="docker run --rm"
    if [[ "$USE_GPU" == true ]]; then
        TRAIN_CMD="$TRAIN_CMD --gpus all"
    fi
    TRAIN_CMD="$TRAIN_CMD -v $PROJECT_ROOT:/workspace/bevnext-sam2"
    TRAIN_CMD="$TRAIN_CMD -v $ABS_DATA_PATH:/workspace/data/nuscenes:ro"
    TRAIN_CMD="$TRAIN_CMD -v $OUTPUT_DIR:/workspace/outputs"
    TRAIN_CMD="$TRAIN_CMD -w /workspace/bevnext-sam2"
    TRAIN_CMD="$TRAIN_CMD -e NUSCENES_DATA_ROOT=/workspace/data/nuscenes"
    TRAIN_CMD="$TRAIN_CMD -e PYTHONPATH=/workspace/bevnext-sam2"
    TRAIN_CMD="$TRAIN_CMD $IMAGE_NAME"
    
    # Build training arguments
    TRAIN_ARGS="python training/train_bevnext_sam2_nuscenes.py --data-root /workspace/data/nuscenes --epochs $NUM_EPOCHS --batch-size $BATCH_SIZE --mixed-precision"
    if [[ -n "$TRAINING_CONFIG" ]]; then
        TRAIN_ARGS="$TRAIN_ARGS --config $TRAINING_CONFIG"
    fi
    
    TRAIN_CMD="$TRAIN_CMD bash -c \"$TRAIN_ARGS\""
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "$TRAIN_CMD"
    else
        eval "$TRAIN_CMD"
        if [[ $? -eq 0 ]]; then
            log_step "Training completed successfully!"
        else
            log_error "Training failed!"
            exit 1
        fi
    fi
}

# Evaluate model
evaluate_model() {
    if [[ "$SKIP_EVAL" == true ]]; then
        log_step "Skipping model evaluation (--skip-eval specified)"
        return 0
    fi
    
    log_step "Evaluating trained model..."
    
    # Find latest checkpoint
    CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints/latest.pth"
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
        log_warning "No checkpoint found at $CHECKPOINT_PATH"
        log_step "Looking for other checkpoints..."
        
        # Look for any checkpoint file
        CHECKPOINT_FILES=($(find "$OUTPUT_DIR" -name "*.pth" -type f 2>/dev/null))
        if [[ ${#CHECKPOINT_FILES[@]} -gt 0 ]]; then
            CHECKPOINT_PATH="${CHECKPOINT_FILES[0]}"
            log_step "Using checkpoint: $CHECKPOINT_PATH"
        else
            log_error "No checkpoint files found!"
            return 1
        fi
    fi
    
    # Build evaluation command
    EVAL_CMD="docker run --rm"
    if [[ "$USE_GPU" == true ]]; then
        EVAL_CMD="$EVAL_CMD --gpus all"
    fi
    EVAL_CMD="$EVAL_CMD -v $PROJECT_ROOT:/workspace/bevnext-sam2"
    EVAL_CMD="$EVAL_CMD -v $ABS_DATA_PATH:/workspace/data/nuscenes:ro"
    EVAL_CMD="$EVAL_CMD -v $OUTPUT_DIR:/workspace/outputs"
    EVAL_CMD="$EVAL_CMD -w /workspace/bevnext-sam2"
    EVAL_CMD="$EVAL_CMD -e NUSCENES_DATA_ROOT=/workspace/data/nuscenes"
    EVAL_CMD="$EVAL_CMD -e PYTHONPATH=/workspace/bevnext-sam2"
    EVAL_CMD="$EVAL_CMD $IMAGE_NAME"
    EVAL_CMD="$EVAL_CMD python validation/validate_model.py --checkpoint $CHECKPOINT_PATH --data-root /workspace/data/nuscenes --output-dir /workspace/outputs/validation --run-nuscenes-eval --generate-viz"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "$EVAL_CMD"
    else
        eval "$EVAL_CMD"
        if [[ $? -eq 0 ]]; then
            log_step "Model evaluation completed!"
        else
            log_error "Model evaluation failed!"
            exit 1
        fi
    fi
}

# Generate final report
generate_report() {
    log_step "Generating pipeline report..."
    
    REPORT_FILE="$OUTPUT_DIR/pipeline_report.txt"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Report would be generated at: $REPORT_FILE"
        return 0
    fi
    
    cat > "$REPORT_FILE" << EOF
BEVNeXt-SAM2 Pipeline Report
============================
Generated: $(date)

Configuration:
- Dataset path: $DATA_PATH
- GPU enabled: $USE_GPU
- Training epochs: $NUM_EPOCHS
- Batch size: $BATCH_SIZE
- Output directory: $OUTPUT_DIR

Pipeline Steps:
$(if [[ "$SKIP_BUILD" == true ]]; then echo "- Container build: SKIPPED"; else echo "- Container build: COMPLETED"; fi)
$(if [[ "$SKIP_VALIDATION" == true ]]; then echo "- Dataset validation: SKIPPED"; else echo "- Dataset validation: COMPLETED"; fi)
$(if [[ "$SKIP_TRAINING" == true ]]; then echo "- Model training: SKIPPED"; else echo "- Model training: COMPLETED"; fi)
$(if [[ "$SKIP_EVAL" == true ]]; then echo "- Model evaluation: SKIPPED"; else echo "- Model evaluation: COMPLETED"; fi)

Output Files:
- Training logs: $OUTPUT_DIR/logs/
- Model checkpoints: $OUTPUT_DIR/checkpoints/
- Validation results: $OUTPUT_DIR/validation/
- Dataset validation: $OUTPUT_DIR/validation_reports/

Next Steps:
1. Review training logs for performance metrics
2. Check validation results for model quality
3. Use trained model for inference
4. Consider hyperparameter tuning if needed

EOF
    
    log_step "Pipeline report generated: $REPORT_FILE"
}

# Main pipeline
main() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}BEVNeXt-SAM2 Complete Pipeline${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    # Check GPU
    check_gpu
    
    # Run pipeline steps
    build_container
    validate_dataset
    train_model
    evaluate_model
    generate_report
    
    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Pipeline completed successfully!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "${CYAN}Results available in: $OUTPUT_DIR${NC}"
    echo -e "${CYAN}Check the pipeline report for details${NC}"
}

# Run main function
main 