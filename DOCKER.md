# Docker Setup for BEVNeXt-SAM2

This document provides comprehensive instructions for running BEVNeXt-SAM2 in Docker containers.

## Prerequisites

### System Requirements
- Docker Engine 20.10+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA 12.1+ support
- At least 16GB RAM
- 50GB+ free disk space

### Installation

1. **Install Docker**:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

2. **Install NVIDIA Docker runtime**:
   ```bash
   # Add NVIDIA package repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   # Install nvidia-docker2
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Verify GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
   ```

## Quick Start

### Method 1: Using Helper Scripts (Recommended)

1. **Build the image**:
   ```bash
   ./scripts/build.sh
   ```

2. **Run the demo**:
   ```bash
   ./scripts/run.sh demo
   ```

3. **Start development environment**:
   ```bash
   ./scripts/run.sh dev
   ```

### Method 2: Using Docker Compose

1. **Build and run the main application**:
   ```bash
   docker-compose up --build bevnext-sam2
   ```

2. **Start development environment with Jupyter**:
   ```bash
   docker-compose up --build dev
   ```

3. **Run specific services**:
   ```bash
   # Training environment
   docker-compose up train
   
   # Inference environment
   docker-compose up inference
   
   # TensorBoard
   docker-compose up tensorboard
   ```

## Docker Images

### Base Image
The Dockerfile uses `nvidia/cuda:12.1-devel-ubuntu22.04` as the base image, which provides:
- Ubuntu 22.04 LTS
- CUDA 12.1 development environment
- cuDNN libraries
- Build tools and compilers

### Image Contents
- Python 3.10
- PyTorch 2.5.1 with CUDA support
- All BEVNeXt-SAM2 dependencies
- Pre-compiled CUDA extensions
- Jupyter Lab
- TensorBoard

## Usage Examples

### 1. Demo Mode
```bash
# Run the fusion demo
./scripts/run.sh demo

# Or with docker-compose
docker-compose up bevnext-sam2
```

### 2. Development Mode
```bash
# Start Jupyter Lab on port 8888
./scripts/run.sh dev

# Start on custom port
./scripts/run.sh dev --port 8889

# Or with docker-compose
docker-compose up dev
```
Access Jupyter at: http://localhost:8888

### 3. Interactive Shell
```bash
# Get a shell inside the container
./scripts/run.sh shell

# Or with docker
docker-compose run --rm bevnext-sam2 bash
```

### 4. Training
```bash
# Start training environment
./scripts/run.sh train

# With specific GPU
./scripts/run.sh train --gpu 1

# Or with docker-compose
docker-compose up train
docker-compose exec train bash
```

### 5. Custom Data and Checkpoints
```bash
# Use custom data directory
./scripts/run.sh demo --data /path/to/your/data --checkpoints /path/to/checkpoints

# Mount additional volumes
docker run -it --rm --gpus all \
  -v /your/data:/workspace/data \
  -v /your/checkpoints:/workspace/checkpoints \
  bevnext-sam2:latest bash
```

## File Structure in Container

```
/workspace/
├── bevnext-sam2/          # Project code
│   ├── bevnext/           # BEVNeXt module
│   ├── sam2_module/       # SAM2 module
│   ├── integration/       # Integration code
│   ├── examples/          # Demo scripts
│   └── configs/           # Configuration files
├── data/                  # Datasets (mounted)
├── checkpoints/           # Model weights (mounted)
├── outputs/               # Results (mounted)
└── logs/                  # Training logs (mounted)
```

## Customization

### 1. Building Custom Images

Modify the Dockerfile and rebuild:
```bash
./scripts/build.sh --tag custom-tag --no-cache
```

### 2. Adding Dependencies

Add to `requirements.txt` or modify the Dockerfile:
```dockerfile
RUN pip install your-additional-package
```

### 3. Custom Entry Points

Create a custom docker-compose service:
```yaml
your-service:
  build: .
  command: python your_custom_script.py
  volumes:
    - ./your_data:/workspace/data
```

## Performance Optimization

### 1. GPU Selection
```bash
# Use specific GPU
./scripts/run.sh demo --gpu 1

# Use multiple GPUs
docker run --gpus '"device=0,1"' bevnext-sam2:latest
```

### 2. Memory Settings
```bash
# Increase shared memory for data loading
docker run --shm-size=32g bevnext-sam2:latest
```

### 3. CPU Optimization
```bash
# Set CPU limits
docker run --cpus="8.0" bevnext-sam2:latest
```

## Troubleshooting

### Common Issues

1. **CUDA not available**:
   ```bash
   # Check NVIDIA Docker runtime
   docker info | grep nvidia
   
   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
   ```

2. **Out of memory**:
   ```bash
   # Increase shared memory
   docker run --shm-size=16g bevnext-sam2:latest
   
   # Use CPU-only mode
   docker run -e CUDA_VISIBLE_DEVICES="" bevnext-sam2:latest
   ```

3. **Permission issues**:
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER ./data ./checkpoints ./outputs
   ```

4. **Build failures**:
   ```bash
   # Clean build cache
   docker system prune -a
   
   # Build without cache
   ./scripts/build.sh --no-cache
   ```

### Debug Mode

Run container with debug information:
```bash
docker run -it --rm --gpus all \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e TORCH_SHOW_CPP_STACKTRACES=1 \
  bevnext-sam2:latest bash
```

## Security Considerations

1. **Non-root user**: The container runs as user `bevnext` (UID 1000)
2. **Read-only filesystem**: Consider using `--read-only` for production
3. **Network isolation**: Use custom networks for multi-container setups
4. **Secret management**: Use Docker secrets for sensitive data

## Production Deployment

### 1. Multi-stage Builds
Create optimized production images:
```dockerfile
FROM bevnext-sam2:latest as production
RUN rm -rf /workspace/bevnext-sam2/.git
RUN pip uninstall -y jupyter matplotlib
```

### 2. Container Orchestration
Use with Kubernetes, Docker Swarm, or other orchestrators:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bevnext-sam2
spec:
  replicas: 1
  selector:
    matchLabels:
      app: bevnext-sam2
  template:
    spec:
      containers:
      - name: bevnext-sam2
        image: bevnext-sam2:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Monitoring

### 1. Container Health
```bash
# Check container health
docker ps
docker logs bevnext-sam2-app

# Monitor resource usage
docker stats bevnext-sam2-app
```

### 2. GPU Monitoring
```bash
# Inside container
watch nvidia-smi

# From host
docker exec bevnext-sam2-app nvidia-smi
```

### 3. TensorBoard
```bash
# Start TensorBoard service
docker-compose up tensorboard

# Access at http://localhost:6006
```

## Support

For Docker-related issues:
1. Check the troubleshooting section above
2. Review Docker and NVIDIA Docker documentation
3. Open an issue with system information:
   ```bash
   docker version
   nvidia-smi
   docker info | grep nvidia
   ``` 