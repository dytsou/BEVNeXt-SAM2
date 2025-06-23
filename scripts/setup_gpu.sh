#!/bin/bash

# BEVNeXt-SAM2 GPU Setup Script
# Comprehensive GPU environment setup with fallback options

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  BEVNeXt-SAM2 GPU Setup Script       ${NC}"
echo -e "${BLUE}======================================${NC}"

# Function to check if running in a container
check_container() {
    if [ -f /.dockerenv ] || grep -q 'docker\|lxc' /proc/1/cgroup 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to get OS information
get_os_info() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
        DISTRO=$ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VER=$(lsb_release -sr)
        DISTRO=$(echo $OS | tr '[:upper:]' '[:lower:]')
    else
        OS=$(uname -s)
        VER=$(uname -r)
        DISTRO="unknown"
    fi
}

# Function to check NVIDIA drivers
check_nvidia_drivers() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA drivers detected${NC}"
        nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | head -3
        return 0
    else
        echo -e "${RED}❌ NVIDIA drivers not found${NC}"
        return 1
    fi
}

# Function to install NVIDIA Docker (Ubuntu/Debian)
install_nvidia_docker_ubuntu() {
    echo -e "${BLUE}Installing NVIDIA Docker for Ubuntu/Debian...${NC}"
    
    # Update package list
    sudo apt update
    
    # Install prerequisites
    sudo apt install -y curl gnupg lsb-release
    
    # Add NVIDIA Docker repository (new method for modern systems)
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Update package list with new repository
    sudo apt update
    
    # Install NVIDIA Container Toolkit
    sudo apt install -y nvidia-container-toolkit
    
    # Configure Docker daemon
    sudo nvidia-ctk runtime configure --runtime=docker
    
    # Restart Docker
    sudo systemctl restart docker
    
    echo -e "${GREEN}✓ NVIDIA Docker installation completed${NC}"
}

# Function to install NVIDIA Docker (CentOS/RHEL)
install_nvidia_docker_centos() {
    echo -e "${BLUE}Installing NVIDIA Docker for CentOS/RHEL...${NC}"
    
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
        sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
    
    sudo yum install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo -e "${GREEN}✓ NVIDIA Docker installation completed${NC}"
}

# Function to test GPU access in Docker
test_gpu_docker() {
    echo -e "${BLUE}Testing GPU access in Docker...${NC}"
    
    # Test with a simple CUDA container
    if docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi; then
        echo -e "${GREEN}✓ GPU access in Docker working!${NC}"
        return 0
    else
        echo -e "${RED}❌ GPU access in Docker failed${NC}"
        return 1
    fi
}

# Function to create CPU fallback configuration
create_cpu_fallback() {
    echo -e "${YELLOW}Creating CPU fallback configuration...${NC}"
    
    cat > training/config_cpu_optimized.json << EOF
{
    "d_model": 128,
    "nhead": 4,
    "num_transformer_layers": 3,
    "num_classes": 10,
    "num_queries": 300,
    "bev_size": [32, 32],
    "image_size": [224, 224],
    "num_cameras": 6,
    "batch_size": 1,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "num_samples": {
        "train": 100,
        "val": 20
    },
    "output_dir": "/workspace/outputs/training_cpu"
}
EOF
    
    echo -e "${GREEN}✓ CPU-optimized config created: training/config_cpu_optimized.json${NC}"
}

# Function to show alternative installation methods
show_alternatives() {
    echo -e "${YELLOW}Alternative Installation Methods:${NC}"
    echo ""
    echo -e "${BLUE}Method 1: Manual NVIDIA Docker Installation${NC}"
    echo "# For Ubuntu 20.04/22.04:"
    echo "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "echo 'deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/ubuntu20.04/$(ARCH) /' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "sudo apt update && sudo apt install -y nvidia-container-toolkit"
    echo "sudo nvidia-ctk runtime configure --runtime=docker"
    echo "sudo systemctl restart docker"
    echo ""
    echo -e "${BLUE}Method 2: Use Pre-built GPU Docker Image${NC}"
    echo "# Pull and use an existing GPU-enabled image:"
    echo "docker pull nvcr.io/nvidia/pytorch:23.10-py3"
    echo "# Then modify our Dockerfile to use this as base image"
    echo ""
    echo -e "${BLUE}Method 3: Cloud GPU Instance${NC}"
    echo "# Use cloud providers with pre-configured GPU environments:"
    echo "# - AWS EC2 P3/P4 instances with Deep Learning AMI"
    echo "# - Google Cloud Platform with GPU-enabled VMs"
    echo "# - Azure with NC-series VMs"
    echo "# - Paperspace, RunPod, or similar GPU cloud services"
}

# Main execution
main() {
    echo -e "${BLUE}Checking system environment...${NC}"
    
    # Get OS information
    get_os_info
    echo -e "OS: $OS $VER ($DISTRO)"
    
    # Check if running in container
    if check_container; then
        echo -e "${YELLOW}⚠ Running inside a container${NC}"
        echo -e "NVIDIA Docker setup should be done on the host system"
        create_cpu_fallback
        echo ""
        echo -e "${YELLOW}For GPU training, please run this script on the host system${NC}"
        show_alternatives
        return 0
    fi
    
    # Check NVIDIA drivers
    if ! check_nvidia_drivers; then
        echo -e "${RED}Please install NVIDIA drivers first:${NC}"
        echo "# Ubuntu/Debian:"
        echo "sudo apt update"
        echo "sudo apt install -y nvidia-driver-470"  # or latest version
        echo "sudo reboot"
        echo ""
        echo "# Check driver installation after reboot:"
        echo "nvidia-smi"
        create_cpu_fallback
        show_alternatives
        return 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found. Please install Docker first:${NC}"
        echo "curl -fsSL https://get.docker.com -o get-docker.sh"
        echo "sudo sh get-docker.sh"
        echo "sudo usermod -aG docker \$USER"
        echo "# Log out and back in, then run this script again"
        return 1
    fi
    
    # Check if NVIDIA Docker is already installed
    if docker info 2>/dev/null | grep -q nvidia; then
        echo -e "${GREEN}✓ NVIDIA Docker runtime already installed${NC}"
        test_gpu_docker
        return 0
    fi
    
    # Install NVIDIA Docker based on distribution
    case $DISTRO in
        ubuntu|debian)
            if install_nvidia_docker_ubuntu; then
                test_gpu_docker
            else
                echo -e "${RED}Failed to install NVIDIA Docker${NC}"
                create_cpu_fallback
                show_alternatives
                return 1
            fi
            ;;
        centos|rhel|fedora)
            if install_nvidia_docker_centos; then
                test_gpu_docker
            else
                echo -e "${RED}Failed to install NVIDIA Docker${NC}"
                create_cpu_fallback
                show_alternatives
                return 1
            fi
            ;;
        *)
            echo -e "${RED}Unsupported distribution: $DISTRO${NC}"
            create_cpu_fallback
            show_alternatives
            return 1
            ;;
    esac
    
    echo -e "${GREEN}🎉 GPU setup completed successfully!${NC}"
    echo -e "${BLUE}You can now run GPU training with:${NC}"
    echo -e "./scripts/train_gpu.sh"
}

# Run main function
main "$@"