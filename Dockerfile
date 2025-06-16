# BEVNeXt-SAM2 Docker Image
# This Dockerfile creates a containerized environment for running BEVNeXt-SAM2

# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (to avoid conflicts)
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /workspace/bevnext-sam2

# Copy requirements and setup files first (for better Docker caching)
COPY setup.py requirements.txt* ./
COPY sam2_module/csrc/ ./sam2_module/csrc/
COPY bevnext/ops/ ./bevnext/ops/

# Install Python dependencies
RUN pip install --no-cache-dir \
    hydra-core>=1.3.2 \
    iopath>=0.1.10 \
    pillow>=9.4.0 \
    tqdm>=4.66.1 \
    numpy>=1.24.4 \
    scikit-image \
    tensorboard \
    matplotlib>=3.9.1 \
    jupyter>=1.0.0 \
    opencv-python>=4.7.0 \
    networkx \
    numba>=0.55.0 \
    trimesh \
    plyfile

# Copy the entire project
COPY . .

# Install the project in development mode
# First, let's try installing the dependencies individually to debug
RUN pip install --no-cache-dir numba>=0.55.0 networkx>=2.2,<2.3
RUN pip install -e . --no-deps

# Create directories for data and checkpoints
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/outputs

# Set environment variables for the application
ENV PYTHONPATH=/workspace/bevnext-sam2:$PYTHONPATH

# Create a non-root user
RUN useradd -m -u 1000 -s /bin/bash bevnext && \
    chown -R bevnext:bevnext /workspace

# Switch to non-root user
USER bevnext

# Expose ports (for Jupyter, TensorBoard, etc.)
EXPOSE 8888 6006

# Set the default command
CMD ["python", "examples/demo_fusion.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import sam2_module; import bevnext; print('All modules loaded successfully')" || exit 1 