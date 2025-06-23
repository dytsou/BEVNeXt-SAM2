# BEVNeXt-SAM2 Complete Dependency Fix

## Overview
This document summarizes the complete dependency fix for BEVNeXt-SAM2, addressing multiple compatibility issues between PyTorch, CUDA, MMCV, and MMDetection.

## Issues Identified

### 1. Initial Issue
- **Error**: `MMCV==2.1.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.1.0`
- **Cause**: MMDet 3.0.0 requires MMCV <2.1.0, but MMCV 2.1.0 was being installed

### 2. Secondary Issue
- **Error**: `The detected CUDA version (11.8) mismatches the version that was used to compile PyTorch (12.6)`
- **Cause**: PyTorch binary was compiled with different CUDA version than the base image

### 3. Root Causes
1. Version incompatibility between PyTorch 2.5.x and MMCV 2.0.x
2. CUDA version mismatch between base image and PyTorch binaries
3. Incorrect PyTorch index URL providing wrong binaries
4. Project setup.py requiring PyTorch >=2.5.1 which conflicts with MMCV

## Complete Solution

### 1. Base Image Update
```dockerfile
# Changed from:
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# To:
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
```
- Added cudnn8 for deep learning support
- Aligned with PyTorch 2.0.0 CUDA requirements

### 2. PyTorch Installation
```dockerfile
# Install PyTorch 2.0.0 with CUDA 11.8
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
- Downgraded from PyTorch 2.5.1 to 2.0.0
- Used correct index URL for CUDA 11.8 binaries
- Added torchaudio for completeness

### 3. MMCV Installation
```dockerfile
# Install MMCV 2.0.0 from pre-built wheels
RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
```
- Used pre-built wheels to avoid compilation issues
- Specified exact torch version in URL (torch2.0.0)

### 4. Environment Variables
```dockerfile
# Set CUDA architecture list for common GPUs
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
- Fixed TORCH_CUDA_ARCH_LIST format (semicolons instead of spaces)
- Added CUDA_HOME and LD_LIBRARY_PATH for proper CUDA detection

### 5. Project Files Update
Updated `setup.py` and `pyproject.toml`:
- Changed `torch>=2.5.1` to `torch>=2.0.0`
- Changed `torchvision>=0.20.1` to `torchvision>=0.15.1`

## Final Version Compatibility Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Base Image | nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 | With cuDNN support |
| Python | 3.10 | From Ubuntu 22.04 |
| CUDA | 11.8 | Toolkit and runtime |
| PyTorch | 2.0.0+cu118 | With CUDA 11.8 support |
| Torchvision | 0.15.1+cu118 | Matches PyTorch |
| MMCV | 2.0.0 | Pre-built for cu118/torch2.0.0 |
| MMEngine | 0.7.1 | Compatible version |
| MMDet | 3.0.0 | As required |
| MMDet3D | 1.4.0 | As required |
| MMSegmentation | 1.0.0 | As required |

## Build Instructions

1. **Build the Docker image**:
   ```bash
   ./scripts/build_fixed.sh
   ```
   This script includes:
   - Prerequisite checks
   - Old image cleanup
   - Build with progress display
   - Verification of the build

2. **Run the container**:
   ```bash
   ./scripts/run_fixed.sh demo
   ```

## Verification
The build script automatically verifies:
- Image creation
- Python imports (torch, mmcv, mmdet)
- CUDA availability
- Version compatibility

## Key Learnings

1. **Version Alignment is Critical**: All components (CUDA, PyTorch, MMCV) must be aligned
2. **Pre-built Wheels**: Using pre-built wheels avoids compilation issues
3. **Index URLs Matter**: PyTorch index URLs must match the exact CUDA version
4. **Environment Variables**: Proper CUDA environment setup is essential for extensions
5. **Dependency Order**: Install order matters - PyTorch first, then MMCV, then MM packages

## Troubleshooting

If you still encounter issues:

1. **Clear Docker cache**:
   ```bash
   docker system prune -a
   docker builder prune -a
   ```

2. **Check CUDA compatibility**:
   ```bash
   nvidia-smi  # Check your GPU and driver version
   ```

3. **Verify installations inside container**:
   ```bash
   docker run --rm bevnext-sam2:latest python -c "
   import torch, mmcv, mmdet
   print(f'Torch: {torch.__version__}')
   print(f'CUDA: {torch.version.cuda}')
   print(f'MMCV: {mmcv.__version__}')
   print(f'MMDet: {mmdet.__version__}')
   "
   ```

This complete fix ensures all dependencies are properly aligned and compatible. 