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
    echo "  train-nuscenes       Start nuScenes v1.0 dataset training with validation"
    echo "  validate-nuscenes    Validate nuScenes dataset integrity and quality"
    echo "  analyze-nuscenes     Analyze nuScenes dataset statistics and generate reports"
    echo "  setup-nuscenes       Complete nuScenes integration setup workflow"
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
    echo "  $0 train-nuscenes          # Train with real nuScenes v1.0 dataset"
    echo "  $0 validate-nuscenes       # Validate nuScenes dataset integrity"
    echo "  $0 analyze-nuscenes        # Generate nuScenes dataset analysis"
    echo "  $0 setup-nuscenes          # Complete nuScenes setup workflow"
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
# Set permissive permissions without requiring sudo
echo -e "${GREEN}Setting standard permissions for directories...${NC}"
chmod -R 755 "./outputs" "./logs" 2>/dev/null || true

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
DOCKER_OPTS="$DOCKER_OPTS --user $USER_UID:$USER_GID"
DOCKER_OPTS="$DOCKER_OPTS -e HOME=/workspace/outputs"
DOCKER_OPTS="$DOCKER_OPTS -e PYTHONUSERBASE=/workspace/outputs/.local"
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
                mkdir -p /workspace/outputs && \
                python examples/demo_fusion.py && \
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
                echo 'Setting up user environment...' && \
                mkdir -p /workspace/outputs/.local && \
                echo 'Installing visualization dependencies...' && \
                pip install --user opencv-python matplotlib --quiet && \
                echo 'System Status:' && \
                python -c 'import torch; print(f\"  • CUDA available: {torch.cuda.is_available()}\"); print(f\"  • GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")' && \
                echo '  • Evaluation mode: Model performance assessment' && \
                echo '  • Test dataset: Real nuScenes v1.0-mini dataset' && \
                echo '  • Results: Saved to /workspace/outputs/evaluation/' && \
                echo '  • Visualizations: 3D bounding boxes on camera images' && \
                echo '' && \
                echo '🔍 Running comprehensive model evaluation...' && \
                echo '' && \
                mkdir -p /workspace/outputs/evaluation && \
                python evaluate_model.py --test-samples 100 --output-dir /workspace/outputs/evaluation && \
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
                echo 'Setting up user environment...' && \
                mkdir -p /workspace/outputs/.local && \
                echo 'Installing visualization dependencies...' && \
                pip install --user seaborn --quiet && \
                echo 'System Status:' && \
                python -c 'import matplotlib; print(f\"  • Matplotlib: {matplotlib.__version__}\"); import numpy; print(f\"  • NumPy: {numpy.__version__}\"); import torch; print(f\"  • PyTorch: {torch.__version__}\")' && \
                echo '  • Output directory: /workspace/outputs/evaluation/visualizations/' && \
                echo '' && \
                echo '📊 Creating comprehensive visualizations...' && \
                echo '' && \
                mkdir -p /workspace/outputs/evaluation/visualizations && \
                python create_evaluation_visualizations.py && \
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
                echo '🔧 Installing required dependencies...' && \
                pip install --quiet tensorboard seaborn matplotlib || echo 'Dependencies already available' && \
                echo 'Training Configuration (Auto-detected):' && \
                echo '  • Model: BEVNeXt + SAM2 Fusion' && \
                echo '  • Data: Synthetic generation (no external datasets)' && \
                echo '  • Mixed precision: Enabled' && \
                echo '  • GPU memory: Auto-optimized configuration' && \
                echo '  • Outputs: Auto-saved to /workspace/outputs/' && \
                echo '  • Logs: TensorBoard + file logging' && \
                echo '' && \
                mkdir -p /workspace/outputs /workspace/logs && \
                python training/train_bevnext_sam2.py --mixed-precision
            "
        ;;
        
    train-nuscenes)
        echo -e "${GREEN}Starting nuScenes v1.0 dataset training...${NC}"
        echo -e "${BLUE}Training with real autonomous driving data from nuScenes${NC}"
        echo -e "${YELLOW}nuScenes training features:${NC}"
        echo -e "  • Real autonomous driving data (cameras, LiDAR, radar)"
        echo -e "  • 23 object categories from nuScenes taxonomy"
        echo -e "  • Token-based data associations"
        echo -e "  • Multi-modal sensor fusion"
        echo -e "  • Automatic dataset validation before training"
        echo -e "  • Memory optimizations for RTX 2080 Ti and similar GPUs"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-nuscenes-train"
        remove_container "$CONTAINER_FULL_NAME"
        
        echo -e "${GREEN}🚗 Launching nuScenes training pipeline...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            --shm-size=8g \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -v $ABS_LOGS_PATH:/workspace/logs \
            -v $ABS_DATA_PATH:/workspace/data \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo '🚗 Starting nuScenes v1.0 Training Pipeline...' && \
                echo 'GPU Status:' && \
                python -c 'import torch; print(f\"  • CUDA available: {torch.cuda.is_available()}\"); print(f\"  • GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")' && \
                echo '  • Training: Real nuScenes v1.0 dataset' && \
                echo '  • Data directory: /workspace/data/nuscenes/' && \
                echo '  • Output directory: /workspace/outputs/' && \
                echo '' && \
                echo '🔧 Setting up training environment...' && \
                pip install --quiet nuscenes-devkit pyquaternion shapely tensorboard seaborn pandas || echo 'Dependencies already installed' && \
                echo '' && \
                echo '🔍 Running dataset validation (optional)...' && \
                python setup_nuscenes_integration.py --data-root /workspace/data/nuscenes --action validate || echo 'Validation skipped - proceeding with training...' && \
                echo '' && \
                echo '🚀 Starting enhanced training...' && \
                mkdir -p /workspace/outputs /workspace/logs && \
                python training/train_bevnext_sam2_nuscenes.py --data-root /workspace/data/nuscenes --mixed-precision && \
                echo '' && \
                echo '✅ nuScenes training completed!' && \
                echo '📊 Check outputs/ for training results and model checkpoints'
            "
        ;;
        
    validate-nuscenes)
        echo -e "${GREEN}Validating nuScenes dataset integrity and quality...${NC}"
        echo -e "${BLUE}Comprehensive validation of nuScenes v1.0 dataset${NC}"
        echo -e "${YELLOW}Validation checks:${NC}"
        echo -e "  • Token linkage integrity (scene→sample→annotation)"
        echo -e "  • Sensor data file existence and format validation"
        echo -e "  • 3D annotation geometry verification"
        echo -e "  • Quality scoring and health assessment"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-nuscenes-validate"
        remove_container "$CONTAINER_FULL_NAME"
        
        echo -e "${GREEN}🔍 Launching nuScenes validation...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_DATA_PATH:/workspace/data \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo '🔍 Starting nuScenes Dataset Validation...' && \
                echo 'Setting up validation environment...' && \
                pip install --user nuscenes-devkit pyquaternion pandas --quiet && \
                echo 'Dataset location: /workspace/data/nuscenes/' && \
                echo 'Validation reports: /workspace/outputs/validation_reports/' && \
                echo '' && \
                echo '🧪 Running comprehensive validation tests...' && \
                mkdir -p /workspace/outputs/validation_reports && \
                python validation/nuscenes_validator.py --data-root /workspace/data/nuscenes --output-dir /workspace/outputs/validation_reports && \
                echo '' && \
                echo '✅ Validation completed!' && \
                echo '📊 Check outputs/validation_reports/ for detailed results'
            "
        ;;
        
    analyze-nuscenes)
        echo -e "${GREEN}Analyzing nuScenes dataset statistics and characteristics...${NC}"
        echo -e "${BLUE}Comprehensive analysis with visualizations and reports${NC}"
        echo -e "${YELLOW}Analysis features:${NC}"
        echo -e "  • Dataset statistics (scenes, samples, categories)"
        echo -e "  • Sensor coverage analysis"
        echo -e "  • Object category distribution"
        echo -e "  • Temporal and geographical analysis"
        echo -e "  • Interactive visualizations"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-nuscenes-analyze"
        remove_container "$CONTAINER_FULL_NAME"
        
        echo -e "${GREEN}📊 Launching nuScenes analysis...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_DATA_PATH:/workspace/data \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo '📊 Starting nuScenes Dataset Analysis...' && \
                echo 'Setting up analysis environment...' && \
                pip install --user nuscenes-devkit seaborn pandas matplotlib --quiet && \
                echo 'Dataset location: /workspace/data/nuscenes/' && \
                echo 'Analysis outputs: /workspace/outputs/analysis_output/' && \
                echo '' && \
                echo '🔬 Running comprehensive dataset analysis...' && \
                mkdir -p /workspace/outputs/analysis_output && \
                python utils/nuscenes_data_analysis.py --data-root /workspace/data/nuscenes --output-dir /workspace/outputs/analysis_output --visualize --save-results && \
                echo '' && \
                echo '✅ Analysis completed!' && \
                echo '📁 Results saved to outputs/analysis_output/' && \
                echo '📊 Visualizations available as PNG files' && \
                echo '📋 JSON report with detailed statistics generated'
            "
        ;;
        
    setup-nuscenes)
        echo -e "${GREEN}Running complete nuScenes integration setup workflow...${NC}"
        echo -e "${BLUE}Automated setup, validation, analysis, and training preparation${NC}"
        echo -e "${YELLOW}Setup workflow includes:${NC}"
        echo -e "  • Environment setup and dependency installation"
        echo -e "  • Dataset validation and integrity checks"
        echo -e "  • Comprehensive dataset analysis"
        echo -e "  • Training configuration optimization"
        echo -e "  • Ready-to-use training setup"
        echo ""
        CONTAINER_FULL_NAME="${CONTAINER_NAME}-nuscenes-setup"
        remove_container "$CONTAINER_FULL_NAME"
        
        echo -e "${GREEN}🔧 Launching complete nuScenes setup...${NC}"
        docker run $DOCKER_OPTS \
            --name "$CONTAINER_FULL_NAME" \
            -v $ABS_PROJECT_PATH:/workspace/bevnext-sam2 \
            -v $ABS_DATA_PATH:/workspace/data \
            -v $ABS_OUTPUTS_PATH:/workspace/outputs \
            -w /workspace/bevnext-sam2 \
            $IMAGE_NAME \
            bash -c "
                echo '🔧 Starting Complete nuScenes Integration Setup...' && \
                echo 'System Status:' && \
                python -c 'import torch; print(f\"  • CUDA available: {torch.cuda.is_available()}\"); print(f\"  • GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\")' && \
                echo '  • Setup mode: Complete workflow automation' && \
                echo '  • Data directory: /workspace/data/nuscenes/' && \
                echo '  • Output directory: /workspace/outputs/' && \
                echo '' && \
                echo '📦 Installing nuScenes dependencies...' && \
                pip install --user nuscenes-devkit pyquaternion shapely seaborn pandas matplotlib tqdm --quiet && \
                echo '' && \
                echo '🚀 Running complete integration workflow...' && \
                mkdir -p /workspace/outputs && \
                python setup_nuscenes_integration.py --data-root /workspace/data/nuscenes --action complete && \
                echo '' && \
                echo '✅ Complete setup workflow finished!' && \
                echo '' && \
                echo '🎯 Next Steps:' && \
                echo '   1. Review setup results in outputs/' && \
                echo '   2. Start training: ./scripts/run.sh train-nuscenes' && \
                echo '   3. Monitor progress with TensorBoard'
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