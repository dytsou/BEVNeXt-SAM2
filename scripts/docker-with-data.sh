#!/bin/bash
# Docker data management script for BEVNeXt-SAM2
# Provides multiple strategies for handling the nuScenes dataset

set -e

# Default configuration
IMAGE_NAME="bevnext-sam2"
CONTAINER_NAME="bevnext-sam2-container"
DEFAULT_HOST_DATA_PATH="/path/to/nuscenes"  # Update this to your actual path
DEFAULT_WORKSPACE_DATA_PATH="/workspace/data/nuscenes"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_help() {
    echo -e "${BLUE}BEVNeXt-SAM2 Docker Data Management${NC}"
    echo ""
    echo "Usage: $0 [STRATEGY] [OPTIONS]"
    echo ""
    echo "Strategies:"
    echo "  volume     - Mount local nuScenes directory (recommended)"
    echo "  download   - Download dataset inside container at runtime"
    echo "  symlink    - Create symbolic link to mounted directory"
    echo "  s3         - Stream data from S3 bucket"
    echo "  shared     - Use shared filesystem (NFS/similar)"
    echo ""
    echo "Options:"
    echo "  --data-path PATH    Host path to nuScenes dataset"
    echo "  --container NAME    Container name (default: $CONTAINER_NAME)"
    echo "  --image NAME        Image name (default: $IMAGE_NAME)"
    echo "  --gpu               Enable GPU support"
    echo "  --jupyter          Start with Jupyter notebook"
    echo "  --port PORT        Expose additional port"
    echo "  --dry-run          Show commands without executing"
    echo ""
    echo "Examples:"
    echo "  $0 volume --data-path /data/nuscenes --gpu"
    echo "  $0 download --jupyter"
    echo "  $0 s3 --data-path s3://my-bucket/nuscenes"
}

# Parse arguments
STRATEGY=""
HOST_DATA_PATH=""
USE_GPU=false
START_JUPYTER=false
EXTRA_PORTS=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        volume|download|symlink|s3|shared)
            STRATEGY="$1"
            shift
            ;;
        --data-path)
            HOST_DATA_PATH="$2"
            shift 2
            ;;
        --container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --jupyter)
            START_JUPYTER=true
            shift
            ;;
        --port)
            EXTRA_PORTS="$EXTRA_PORTS -p $2:$2"
            shift 2
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

if [[ -z "$STRATEGY" ]]; then
    echo -e "${RED}Error: Strategy is required${NC}"
    print_help
    exit 1
fi

# Set default data path if not provided
if [[ -z "$HOST_DATA_PATH" ]]; then
    HOST_DATA_PATH="$DEFAULT_HOST_DATA_PATH"
fi

# Build base docker command
DOCKER_CMD="docker run -it --rm"
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"

# Add GPU support if requested
if [[ "$USE_GPU" == true ]]; then
    DOCKER_CMD="$DOCKER_CMD --gpus all"
fi

# Add standard ports
DOCKER_CMD="$DOCKER_CMD -p 8888:8888 -p 6006:6006"

# Add extra ports
if [[ -n "$EXTRA_PORTS" ]]; then
    DOCKER_CMD="$DOCKER_CMD $EXTRA_PORTS"
fi

# Strategy-specific configurations
case $STRATEGY in
    volume)
        echo -e "${GREEN}Using volume mount strategy${NC}"
        if [[ ! -d "$HOST_DATA_PATH" ]]; then
            echo -e "${YELLOW}Warning: Host data path does not exist: $HOST_DATA_PATH${NC}"
            echo "Please ensure the nuScenes dataset is available at this location"
        fi
        DOCKER_CMD="$DOCKER_CMD -v $HOST_DATA_PATH:$DEFAULT_WORKSPACE_DATA_PATH:ro"
        ;;
        
    download)
        echo -e "${GREEN}Using download strategy${NC}"
        echo "Dataset will be downloaded at container startup"
        DOCKER_CMD="$DOCKER_CMD -v nuscenes_data:$DEFAULT_WORKSPACE_DATA_PATH"
        DOCKER_CMD="$DOCKER_CMD -e DOWNLOAD_NUSCENES=true"
        ;;
        
    symlink)
        echo -e "${GREEN}Using symlink strategy${NC}"
        DOCKER_CMD="$DOCKER_CMD -v $HOST_DATA_PATH:/host/nuscenes:ro"
        DOCKER_CMD="$DOCKER_CMD -e CREATE_SYMLINK=true"
        ;;
        
    s3)
        echo -e "${GREEN}Using S3 streaming strategy${NC}"
        DOCKER_CMD="$DOCKER_CMD -e S3_BUCKET_PATH=$HOST_DATA_PATH"
        DOCKER_CMD="$DOCKER_CMD -e USE_S3_STREAM=true"
        echo "Note: Ensure AWS credentials are configured"
        ;;
        
    shared)
        echo -e "${GREEN}Using shared filesystem strategy${NC}"
        DOCKER_CMD="$DOCKER_CMD -v $HOST_DATA_PATH:$DEFAULT_WORKSPACE_DATA_PATH:ro"
        echo "Note: Ensure shared filesystem is properly mounted on host"
        ;;
esac

# Add image
DOCKER_CMD="$DOCKER_CMD $IMAGE_NAME"

# Add startup command
if [[ "$START_JUPYTER" == true ]]; then
    DOCKER_CMD="$DOCKER_CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
else
    DOCKER_CMD="$DOCKER_CMD bash"
fi

echo ""
echo -e "${BLUE}Docker Command:${NC}"
echo "$DOCKER_CMD"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}Dry run mode - not executing${NC}"
    exit 0
fi

# Execute the command
echo -e "${GREEN}Starting container...${NC}"
eval $DOCKER_CMD 