#!/bin/bash

# BEVNeXt-SAM2 Docker Run Script
# This script provides easy access to different ways of running the container

set -e  # Exit on any error

USER=$(whoami)
GROUP=$(id -gn)
USER_UID=$(id -u)
USER_GID=$(id -g)

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
    echo "  demo                 Run automatic BEVNeXt-SAM2 demo (no prompts)"
    echo "  test-model           Run comprehensive testing suite (validates setup)"
    echo "  eval                 Evaluate trained model performance on test dataset"
    echo "  viz, visualize       Generate comprehensive evaluation visualizations"
    echo "  dev                  Start development environment with Jupyter"
    echo "  shell                Start interactive shell in container"
    echo "  train                Start automatic BEVNeXt-SAM2 training (no prompts)"
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
    echo "  $0 demo                    # Run automatic demo (no prompts)"
    echo "  $0 test-model              # Run comprehensive test suite"
    echo "  $0 eval                    # Evaluate trained model performance"
    echo "  $0 viz                     # Generate evaluation visualizations"
    echo "  $0 dev --port 8889         # Start Jupyter on port 8889"
    echo "  $0 shell                   # Interactive shell"
    echo "  $0 train --gpu 1           # Automatic training on GPU 1"
}

# Function to check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^$1$"
}

# Function to remove existing container
remove_container() {
    if container_exists "$1"; then
        echo -e "${YELLOW}Removing existing container: $1${NC}"
        docker rm -f "$1" > /dev/null 2>&1
    fi
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
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo -e "${RED}Error: --gpu requires a value${NC}"
                exit 1
            fi
            GPU_ID="$2"
            shift 2
            ;;
        --port)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo -e "${RED}Error: --port requires a value${NC}"
                exit 1
            fi
            JUPYTER_PORT="$2"
            shift 2
            ;;
        --data)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo -e "${RED}Error: --data requires a value${NC}"
                exit 1
            fi
            DATA_PATH="$2"
            shift 2
            ;;
        --checkpoints)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo -e "${RED}Error: --checkpoints requires a value${NC}"
                exit 1
            fi
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

# Ensure proper permissions for Docker user (UID $USER_UID)
echo -e "${BLUE}Setting directory permissions...${NC}"
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    # Sudo is available and can run without password
    echo -e "${GREEN}Setting ownership to $USER:$GROUP (UID:GID $USER_UID:$USER_GID)...${NC}"
    sudo chown -R $USER:$GROUP "./outputs" "./logs" 2>/dev/null || {
        echo -e "${YELLOW}Warning: Could not change ownership. Setting open permissions...${NC}"
        chmod -R 755 "./outputs" "./logs" 2>/dev/null || true
    }
else
    # No sudo or sudo requires password, use permissive permissions
    echo -e "${YELLOW}Setting open permissions (no sudo access)...${NC}"
    chmod -R 755 "./outputs" "./logs" 2>/dev/null || true
fi

# Get absolute paths after creating directories
ABS_DATA_PATH=$(cd "$DATA_PATH" && pwd)
ABS_CHECKPOINTS_PATH=$(cd "$CHECKPOINTS_PATH" && pwd)
ABS_OUTPUTS_PATH=$(cd "./outputs" && pwd)
ABS_LOGS_PATH=$(cd "./logs" && pwd)
ABS_PROJECT_PATH=$(pwd)

# Check if NVIDIA runtime is available
RUNTIME_OPTS=""
if docker info 2>/dev/null | grep -q nvidia; then
    RUNTIME_OPTS="--runtime=nvidia"
    echo -e "${GREEN}NVIDIA Docker runtime detected${NC}"
else
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected. GPU support may not work.${NC}"
fi

# Common Docker run options
DOCKER_OPTS="--rm -it"
DOCKER_OPTS="$DOCKER_OPTS $RUNTIME_OPTS"
DOCKER_OPTS="$DOCKER_OPTS -e NVIDIA_VISIBLE_DEVICES=all"
DOCKER_OPTS="$DOCKER_OPTS -e CUDA_VISIBLE_DEVICES=$GPU_ID"
DOCKER_OPTS="$DOCKER_OPTS -v $ABS_DATA_PATH:/workspace/data"
DOCKER_OPTS="$DOCKER_OPTS -v $ABS_CHECKPOINTS_PATH:/workspace/checkpoints"
DOCKER_OPTS="$DOCKER_OPTS -v $ABS_OUTPUTS_PATH:/workspace/outputs"
DOCKER_OPTS="$DOCKER_OPTS -v $ABS_LOGS_PATH:/workspace/logs"

# Run based on mode
case $MODE in
    demo)
        echo -e "${GREEN}Running automatic BEVNeXt-SAM2 demo...${NC}"
        echo -e "${BLUE}Demo will showcase 3D detection + 2D segmentation fusion${NC}"
        echo -e "${YELLOW}Demo features:${NC}"
        echo -e "  • BEV-SAM fusion demonstration"
        echo -e "  • SAM-enhanced 3D detection"
        echo -e "  • Synthetic data generation"
        echo -e "  • Real-time inference simulation"
        echo -e "  • Visual output examples"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-demo"
        remove_container "$CONTAINER_FULL_NAME"
        
        echo -e "${GREEN}🎬 Launching BEVNeXt-SAM2 demo...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo '🎯 Starting BEVNeXt-SAM2 Integration Demo...' && \
                echo 'System Status:' && \
                python -c 'import torch; print(f\"  • CUDA available: {torch.cuda.is_available()}\"); print(f\"  • GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")' && \
                echo '  • Demo mode: Automatic execution' && \
                echo '  • Data: Synthetic generation' && \
                echo '  • Outputs: Saved to /workspace/outputs/' && \
                echo '' && \
                echo '🚀 Running fusion demonstrations...' && \
                echo '' && \
                chown -R $USER_UID:$USER_GID /workspace/outputs 2>/dev/null || true && \
                mkdir -p /workspace/outputs && \
                chmod -R 755 /workspace/outputs 2>/dev/null || true && \
                su -s /bin/bash $USER_UID -c 'cd /workspace/bevnext-sam2 && python examples/demo_fusion.py' && \
                echo '' && \
                echo '✅ Demo completed successfully!' && \
                echo '' && \
                echo '📁 Check outputs/ directory for generated results' && \
                echo '🎯 Ready for training with: ./scripts/run.sh train'
            "
        ;;
        
    test-model)
        echo -e "${GREEN}Running comprehensive BEVNeXt-SAM2 test suite...${NC}"
        echo -e "${BLUE}Testing will validate setup, dependencies, and functionality${NC}"
        echo -e "${YELLOW}Test categories:${NC}"
        echo -e "  • Basic imports and dependencies"
        echo -e "  • Project structure validation"
        echo -e "  • Docker setup verification"
        echo -e "  • Training configuration tests"
        echo -e "  • Synthetic data generation"
        echo -e "  • Model creation and integration"
        echo -e "  • Mini training simulation"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-test"
        remove_container "$CONTAINER_FULL_NAME"
        
        echo -e "${GREEN}🧪 Launching comprehensive test suite...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo '🧪 Starting BEVNeXt-SAM2 Comprehensive Test Suite...' && \
                echo 'System Status:' && \
                python -c 'import torch; print(f\"  • CUDA available: {torch.cuda.is_available()}\"); print(f\"  • GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")' && \
                echo '  • Test mode: Comprehensive validation' && \
                echo '  • Outputs: Saved to /workspace/outputs/' && \
                echo '' && \
                echo '🔍 Running all test categories...' && \
                echo '' && \
                python test_training_setup.py && \
                echo '' && \
                echo '✅ Test suite completed!' && \
                echo '' && \
                echo '📊 All tests validate setup readiness for training' && \
                echo '🎯 Ready for training with: ./scripts/run.sh train'
            "
        ;;
        
    eval)
        echo -e "${GREEN}Running BEVNeXt-SAM2 model evaluation...${NC}"
        echo -e "${BLUE}Evaluation will assess trained model performance on test dataset${NC}"
        echo -e "${YELLOW}Evaluation metrics:${NC}"
        echo -e "  • Detection performance (mAP, Precision, Recall)"
        echo -e "  • Segmentation quality (IoU, Dice coefficient)"
        echo -e "  • Inference speed (FPS, latency)"
        echo -e "  • Overall model assessment"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-eval"
        remove_container "$CONTAINER_FULL_NAME"
        
        echo -e "${GREEN}📊 Launching model evaluation...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -v $ABS_CHECKPOINTS_PATH:/workspace/checkpoints \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo '📊 Starting BEVNeXt-SAM2 Model Evaluation...' && \
                echo 'Installing visualization dependencies...' && \
                pip install opencv-python matplotlib --quiet && \
                echo 'System Status:' && \
                python -c 'import torch; print(f\"  • CUDA available: {torch.cuda.is_available()}\"); print(f\"  • GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")' && \
                echo '  • Evaluation mode: Model performance assessment' && \
                echo '  • Test dataset: Real nuScenes v1.0-mini dataset' && \
                echo '  • Results: Saved to /workspace/outputs/evaluation/' && \
                echo '  • Visualizations: 3D bounding boxes on camera images' && \
                echo '' && \
                echo '🔍 Running comprehensive model evaluation...' && \
                echo '' && \
                chown -R $USER_UID:$USER_GID /workspace/outputs 2>/dev/null || true && \
                mkdir -p /workspace/outputs/evaluation && \
                chmod -R 755 /workspace/outputs/evaluation 2>/dev/null || true && \
                su -s /bin/bash $USER_UID -c 'cd /workspace/bevnext-sam2 && python evaluate_model.py --test-samples 100 --output-dir /workspace/outputs/evaluation' && \
                echo '' && \
                echo '✅ Model evaluation completed!' && \
                echo '' && \
                echo '📊 Check outputs/evaluation/ for detailed results' && \
                echo '🎯 Performance metrics and grades available in report' && \
                echo '🎨 3D bounding box visualizations available in sample_visualizations/'
            "
        ;;
        
    viz|visualize)
        echo -e "${GREEN}Creating comprehensive evaluation visualizations...${NC}"
        echo -e "${BLUE}Generating performance charts, sample predictions, and interactive reports${NC}"
        echo -e "${YELLOW}Visualization outputs:${NC}"
        echo -e "  • Performance dashboard with comprehensive metrics"
        echo -e "  • Detailed charts and statistical analyses"
        echo -e "  • Sample prediction visualizations"
        echo -e "  • 3D point cloud and BEV visualizations"
        echo -e "  • Interactive HTML report"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-viz"
        remove_container "$CONTAINER_FULL_NAME"
        
        echo -e "${GREEN}🎨 Launching visualization generator...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -v $ABS_CHECKPOINTS_PATH:/workspace/checkpoints \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo '🎨 Starting BEVNeXt-SAM2 Evaluation Visualization...' && \
                echo 'Installing visualization dependencies...' && \
                pip install seaborn --quiet && \
                echo 'System Status:' && \
                python -c 'import matplotlib; print(f\"  • Matplotlib: {matplotlib.__version__}\"); import numpy; print(f\"  • NumPy: {numpy.__version__}\"); import torch; print(f\"  • PyTorch: {torch.__version__}\")' && \
                echo '  • Output directory: /workspace/outputs/evaluation/visualizations/' && \
                echo '' && \
                echo '📊 Creating comprehensive visualizations...' && \
                echo '' && \
                chown -R $USER_UID:$USER_GID /workspace/outputs 2>/dev/null || true && \
                mkdir -p /workspace/outputs/evaluation/visualizations && \
                chmod -R 755 /workspace/outputs/evaluation 2>/dev/null || true && \
                su -s /bin/bash $USER_UID -c 'cd /workspace/bevnext-sam2 && python create_evaluation_visualizations.py' && \
                echo '' && \
                echo '✅ Visualization generation completed!' && \
                echo '' && \
                echo '📁 Visualizations saved to outputs/evaluation/visualizations/' && \
                echo '🌐 Open evaluation_report.html in your browser for interactive view!' && \
                echo '📊 Performance dashboard and detailed charts available as PNG files'
            "
        ;;
        
    dev)
        echo -e "${GREEN}Starting development environment with Jupyter Lab...${NC}"
        echo -e "${BLUE}Jupyter will be available at: http://localhost:${JUPYTER_PORT}${NC}"
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-dev"
        remove_container "$CONTAINER_FULL_NAME"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -p ${JUPYTER_PORT}:8888 \
            -p 6006:6006 \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                       --NotebookApp.token='' --NotebookApp.password=''
        ;;
        
    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-shell"
        remove_container "$CONTAINER_FULL_NAME"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash
        ;;
        
    train)
        echo -e "${GREEN}Starting automatic BEVNeXt-SAM2 training...${NC}"
        echo -e "${BLUE}Training will start automatically with optimal configuration${NC}"
        echo -e "${YELLOW}Training features:${NC}"
        echo -e "  • Synthetic data generation (no external datasets needed)"
        echo -e "  • Mixed precision training for memory optimization"
        echo -e "  • Automatic checkpointing and logging"
        echo -e "  • TensorBoard monitoring"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-train"
        remove_container "$CONTAINER_FULL_NAME"
        
        # Output directories will be auto-created by training script based on GPU memory
        
        # Run training with automatic configuration
        echo -e "${GREEN}🚀 Launching BEVNeXt-SAM2 training...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            --shm-size=8g \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -v $ABS_LOGS_PATH:/workspace/logs \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo 'Starting BEVNeXt-SAM2 training...' && \
                echo 'GPU Status:' && \
                python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available()); print(\"GPU count:\", torch.cuda.device_count())' && \
                echo '' && \
                echo 'Training Configuration (Auto-detected):' && \
                echo '  • Model: BEVNeXt + SAM2 Fusion' && \
                echo '  • Data: Synthetic generation (no external datasets)' && \
                echo '  • Mixed precision: Enabled' && \
                echo '  • GPU memory: Auto-optimized configuration' && \
                echo '  • Outputs: Auto-saved to /workspace/outputs/' && \
                echo '  • Logs: TensorBoard + file logging' && \
                echo '' && \
                chown -R $USER_UID:$USER_GID /workspace/outputs /workspace/logs 2>/dev/null || true && \
                mkdir -p /workspace/outputs /workspace/logs && \
                chmod -R 755 /workspace/outputs /workspace/logs 2>/dev/null || true && \
                su -s /bin/bash $USER_UID -c 'cd /workspace/bevnext-sam2 && python training/train_bevnext_sam2.py --mixed-precision'
            "
        ;;
        
    inference)
        echo -e "${GREEN}Starting inference environment...${NC}"
        echo -e "${BLUE}Use this environment for running inference${NC}"
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-inference"
        remove_container "$CONTAINER_FULL_NAME"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash
        ;;
        
    tensorboard)
        echo -e "${GREEN}Starting TensorBoard server...${NC}"
        echo -e "${BLUE}TensorBoard will be available at: http://localhost:6006${NC}"
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-tensorboard"
        remove_container "$CONTAINER_FULL_NAME"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
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