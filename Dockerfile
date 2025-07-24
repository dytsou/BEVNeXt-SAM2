# BEVNeXt-SAM2 Docker Image
# This Dockerfile creates a containerized environment for running BEVNeXt-SAM2

# Use CUDA 11.7 base image for compatibility with PyTorch 1.13.1
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
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
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace/bevnext-sam2

# Copy requirements and setup files first (for better Docker caching)
COPY setup.py requirements.txt* ./
COPY sam2_module/csrc/ ./sam2_module/csrc/
COPY bevnext/ops/ ./bevnext/ops/

# Copy constraints file first
COPY constraints.txt ./

# Install NumPy first with the correct version to avoid build issues
RUN pip install --no-cache-dir numpy==1.19.5

# Install core Python dependencies with constraints
# NOTE: Using constraints file to ensure consistent versions
RUN pip install --no-cache-dir --retries 5 --timeout 300 -c constraints.txt \
    hydra-core>=1.3.2 \
    iopath>=0.1.10 \
    pillow>=9.4.0 \
    tqdm>=4.66.1 \
    scikit-image \
    tensorboard \
    seaborn \
    matplotlib>=3.9.1 \
    jupyter>=1.0.0 \
    opencv-python>=4.7.0 \
    numba>=0.55.0 \
    plyfile

# Install trimesh with specific version
RUN pip install --no-cache-dir --retries 5 --timeout 300 "trimesh>=2.35.39,<2.35.40"

# Install networkx separately
RUN pip install --no-cache-dir --retries 5 --timeout 300 "networkx>=2.2,<2.3"

# Install PyTorch 1.13.1 with CUDA 11.7
# Downgrading PyTorch to support MMCV 1.7.0
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install build tools to speed up compilation
RUN pip install --no-cache-dir ninja psutil

# Install openmim to manage OpenMMLab packages
RUN pip install --no-cache-dir openmim

# Install MMCV using mim (will find the best compatible version)
# Installing MMCV 1.7.0 as required by BEVNeXt (expects 1.5.2-1.7.0)
# Note: This will build from source if no pre-built wheel is available (may take 10-20 minutes)
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 mim install mmcv-full==1.7.0

# For MMCV 1.7.0, we need compatible versions of MM packages
# Install mmdet and mmsegmentation first (they don't require CUDA compilation)
RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    mmdet==2.28.2 \
    mmsegmentation==0.30.0

# For mmdet3d, we'll use a workaround to avoid CUDA compilation issues
# Option 1: Try to install without CUDA extensions first
# Option 2: Use a version that has pre-built wheels available
# Setting environment variables to help with CUDA detection
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV FORCE_CUDA=1
ENV MAX_JOBS=2

# Try different approaches to install mmdet3d
RUN pip install --no-cache-dir mmdet3d==1.0.0rc4 || \
    pip install --no-cache-dir mmdet3d==1.0.0rc5 || \
    pip install --no-cache-dir mmdet3d==1.0.0rc6 || \
    (echo "Installing mmdet3d from source with minimal CUDA..." && \
     pip install --no-cache-dir --no-deps mmdet3d==0.17.2 && \
     pip install --no-cache-dir terminaltables pyquaternion fire lyft_dataset_sdk nuscenes-devkit)

# Install remaining dependencies for mmdet packages (excluding mmcv which we already have)
RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    terminaltables \
    pycocotools \
    scipy \
    shapely \
    yapf

# Install nuScenes dependencies
RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    nuscenes-devkit \
    pyquaternion \
    descartes \
    seaborn \
    pandas

# Install additional dependencies
RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    eva-decord>=0.6.1 \
    Flask>=3.0.3 \
    Flask-Cors>=5.0.0 \
    av>=13.0.0

# Copy the entire project
COPY . .

# Ensure nuScenes integration files are properly placed
COPY training/nuscenes_dataset_v2.py ./training/
COPY training/train_bevnext_sam2_nuscenes.py ./training/
COPY validation/nuscenes_validator.py ./validation/
COPY utils/nuscenes_data_analysis.py ./utils/
COPY setup_nuscenes_integration.py ./

# Create validation and utils directories if they don't exist
RUN mkdir -p validation utils

# Copy the minimal setup.py for Docker build

# Set CUDA architecture list for common GPUs
# Covers Volta (V100), Turing (RTX 20xx), Ampere (RTX 30xx, A100), and Ada Lovelace (RTX 40xx)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1

# Set environment variables to ensure CUDA is properly detected
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install the project in development mode with verbose output
# First, let's check if CUDA is properly detected
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Install the project using regular setup (without CUDA extensions)
# This avoids compilation issues during Docker build
RUN FORCE_CUDA=0 pip install --no-deps -e .

# Create directories for data, checkpoints, and nuScenes integration
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/outputs \
    /workspace/data/nuscenes /workspace/outputs/validation_reports \
    /workspace/outputs/analysis_output /workspace/outputs/training_nuscenes

# Copy nuScenes dataset directly into the image
# This assumes the nuScenes dataset is located at ../../nuscenes relative to the Dockerfile
COPY /mnt/HHD1/dytsou/nuscenes /workspace/data/nuscenes

# Set environment variables for the application
ENV PYTHONPATH=/workspace/bevnext-sam2
ENV NUSCENES_DATA_ROOT=/workspace/data/nuscenes

# Create a non-root user
RUN useradd -m -u 1000 -s /bin/bash bevnext && \
    chown -R bevnext:bevnext /workspace && \
    chmod 755 /workspace/data /workspace/outputs

# Switch to non-root user
USER bevnext

# Expose ports (for Jupyter, TensorBoard, etc.)
EXPOSE 8888 6006

# Set the default command
CMD ["python", "examples/demo_fusion.py"]

# Health check (includes nuScenes integration check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import sam2_module; import bevnext; \
                   try: from nuscenes.nuscenes import NuScenes; print('All modules including nuScenes loaded successfully'); \
                   except: print('Core modules loaded, nuScenes integration available')" || exit 1 