# BEVNeXt-SAM2: Unified 3D Object Detection and Segmentation

This repository merges [BEVNeXt](https://github.com/woxihuanjiangguo/BEVNeXt) and [SAM 2](https://github.com/facebookresearch/sam2) to create a unified framework for 3D object detection with enhanced segmentation capabilities.

## 🚀 Quick Start

### Multi-GPU Training (Recommended for Production)
For servers with multiple GPUs, the fastest way to start:

```bash
# Build Docker container
./scripts/run.sh build-fast --gpu

# Multi-GPU training with Docker (recommended)
./scripts/launch_docker_multi_gpu.sh \
    --num-gpus 2 \
    --batch-size 4 \
    --epochs 50 \
    --data-root /path/to/nuscenes

# Or native multi-GPU training
./scripts/launch_distributed_training.sh \
    --num-gpus 2 \
    --batch-size 4 \
    --epochs 50 \
    --data-root /path/to/nuscenes
```

### Single GPU / Development
```bash
# Build and run demo
./scripts/build.sh
./scripts/run.sh demo

# Development environment with Jupyter
./scripts/run.sh dev
```

## Overview

BEVNeXt-SAM2 combines:
- **BEVNeXt**: State-of-the-art Bird's Eye View (BEV) 3D object detection
- **SAM 2**: Segment Anything Model for high-quality image and video segmentation
- **Multi-GPU Training**: Production-ready multi-GPU training with DataParallel and DistributedDataParallel support

The integration enables:
- 3D object detection from multi-view cameras using BEV representation
- Instance segmentation masks for detected objects using SAM 2
- Enhanced perception by combining geometric (3D boxes) and appearance (segmentation) information
- High-performance training on multi-GPU systems with up to 1.95x speedup

## Architecture

```
BEVNeXt-SAM2/
├── bevnext/              # BEVNeXt 3D detection module
├── sam2_module/          # SAM 2 segmentation module  
├── integration/          # Fusion modules combining both
├── training/             # Training scripts and utilities
│   ├── train_bevnext_sam2.py           # Main training script
│   ├── train_bevnext_sam2_nuscenes.py  # nuScenes-specific training
│   ├── multi_gpu_utils.py              # Multi-GPU training utilities
│   └── nuscenes_dataset_v2.py          # Enhanced dataset loader
├── configs/              # Configuration files
│   ├── bevnext/          # BEVNeXt configs
│   └── sam2/             # SAM 2 configs
├── tools/                # Training and inference utilities
├── scripts/              # Docker and utility scripts
│   ├── launch_docker_multi_gpu.sh      # Docker multi-GPU launcher
│   ├── launch_distributed_training.sh  # Native multi-GPU launcher
│   ├── monitor_training.sh             # Training monitoring
│   └── run.sh                          # Main Docker interface
├── validation/           # Model evaluation and testing
├── data/                 # Dataset storage and processing
├── checkpoints/          # Model checkpoints
└── docs/                 # Documentation
```

## Installation

### Option 1: Docker (Recommended)

Docker provides the easiest way to get started with all dependencies pre-installed:

#### Quick Start
```bash
# Build the Docker image
./scripts/build.sh

# Fast GPU-optimized build
./scripts/run.sh build-fast --gpu

# Run the demo
./scripts/run.sh demo

# Start development environment with Jupyter
./scripts/run.sh dev

# Multi-GPU training (production ready)
./scripts/launch_docker_multi_gpu.sh --num-gpus 2 --batch-size 4 --data-root /path/to/data
```

#### Using Docker Compose
```bash
# Run demo
docker-compose up --build bevnext-sam2

# Development with Jupyter Lab
docker-compose up --build dev

# Multi-GPU training
docker-compose up --build train
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

### Option 2: Native Installation

#### Requirements
- Python >= 3.10
- PyTorch >= 2.5.1
- CUDA toolkit (for GPU support)

#### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-repo/bevnext-sam2.git
cd bevnext-sam2
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### BEV-SAM Fusion

The main integration module combines BEVNeXt's 3D detections with SAM 2's segmentation:

```python
from integration import BEVSAMFusion

# Initialize fusion module
fusion = BEVSAMFusion(
    sam2_checkpoint="path/to/sam2_checkpoint.pt",
    sam2_model_cfg="configs/sam2/sam2_hiera_l.yaml"
)

# Run fusion
results = fusion(
    image=image_tensor,
    bev_detections=bev_outputs,
    camera_params=camera_calibration
)

# Results contain:
# - 3D bounding boxes
# - 2D projected boxes
# - Instance segmentation masks
# - Detection scores and labels
```

### Demo Script

Run the demo to see both integration approaches:

```bash
# Native installation
python examples/demo_fusion.py

# Docker
./scripts/run.sh demo
```

### Training

#### Multi-GPU Training (Recommended)

For production environments with multiple GPUs:

```bash
# Docker multi-GPU training (recommended)
./scripts/launch_docker_multi_gpu.sh \
    --num-gpus 2 \
    --batch-size 4 \
    --epochs 50 \
    --data-root /path/to/nuscenes \
    --lr-scaling linear

# Native multi-GPU training
./scripts/launch_distributed_training.sh \
    --num-gpus 2 \
    --batch-size 4 \
    --epochs 50 \
    --data-root /path/to/nuscenes

# Manual distributed training with torchrun
torchrun --nproc_per_node=2 \
    training/train_bevnext_sam2_nuscenes.py \
    --data-root /path/to/nuscenes \
    --gpus 0,1 \
    --distributed \
    --batch-size 4 \
    --mixed-precision
```

#### Single GPU Training

For development and smaller datasets:

```bash
# Native installation
python training/train_bevnext_sam2.py --config configs/bevnext/bevnext-stage1.py
python training/train_bevnext_sam2_nuscenes.py --data-root /path/to/nuscenes

# Docker
./scripts/run.sh train
docker-compose exec train bash
```

#### Training Monitoring

Monitor training progress in real-time:

```bash
# Monitor GPU usage and training metrics
./scripts/monitor_training.sh

# TensorBoard (if configured)
./scripts/run.sh tensorboard
```

### Inference

```bash
# Native installation
python tools/inference.py \
    --config configs/fusion_config.yaml \
    --checkpoint path/to/checkpoint.pth \
    --input path/to/images \
    --output path/to/results

# Docker
./scripts/run.sh inference
```

## Docker Usage

### Available Services

- **demo**: Run the fusion demo
- **dev**: Development environment with Jupyter Lab
- **train**: Single GPU training environment
- **train-multi-gpu**: Multi-GPU training with automatic scaling
- **inference**: Inference environment
- **tensorboard**: TensorBoard monitoring
- **shell**: Interactive development shell

### Examples

```bash
# Quick demo
./scripts/run.sh demo

# Development with Jupyter (http://localhost:8888)
./scripts/run.sh dev

# Interactive shell
./scripts/run.sh shell

# Single GPU training
./scripts/run.sh train --gpu

# Multi-GPU training (production ready)
./scripts/run.sh train-multi-gpu --data-path /path/to/nuscenes --gpu --epochs 50

# Advanced multi-GPU training with custom options
./scripts/launch_docker_multi_gpu.sh \
    --num-gpus 2 \
    --batch-size 4 \
    --lr-scaling linear \
    --gradient-accumulation 2 \
    --mixed-precision

# TensorBoard monitoring
./scripts/run.sh tensorboard

# Training monitoring
./scripts/monitor_training.sh
```

### Docker Compose Services

```bash
# Main application
docker-compose up bevnext-sam2

# Development environment
docker-compose up dev

# Specific services
docker-compose up train
docker-compose up inference
docker-compose up tensorboard
```

For comprehensive Docker documentation, see [DOCKER.md](DOCKER.md).

## Model Zoo

| Model | Task | Config | Checkpoint |
|-------|------|--------|------------|
| BEVNeXt-R50 | 3D Detection | [config](configs/bevnext/bevnext-stage2.py) | [download](#) |
| SAM2-Large | Segmentation | [config](configs/sam2/sam2_hiera_l.yaml) | [download](#) |
| BEVNeXt-SAM2 | Fusion | [config](configs/fusion_config.yaml) | [download](#) |

## Key Features

### 1. 3D-to-2D Projection
Projects 3D bounding boxes from BEV space to 2D image plane for SAM 2 prompting.

### 2. Multi-View Fusion
Handles multiple camera views with proper coordinate transformations.

### 3. Instance Segmentation
Generates high-quality masks for each detected 3D object.

### 4. **Multi-GPU Training** 🚀
- **DataParallel (DP)** and **DistributedDataParallel (DDP)** support
- **Automatic GPU detection** and configuration
- **Learning rate scaling** (linear, sqrt, or none)
- **Mixed precision training** for memory efficiency
- **Up to 1.95x speedup** on multi-GPU systems
- **Gradient accumulation** for large effective batch sizes

### 5. Production-Ready Training
- **Real-time monitoring** with GPU utilization tracking
- **Automatic batch size scaling** based on GPU count
- **Distributed data loading** with proper sampling
- **Training checkpointing** and resume capabilities

### 6. Flexible Architecture
Modular design allows using components independently or together.

### 7. Containerized Deployment
Full Docker support with multiple deployment options and multi-GPU container orchestration.

## Integration Modules

### BEVSAMFusion
Combines BEVNeXt's 3D detections with SAM2's segmentation:
- Projects 3D boxes to 2D image coordinates
- Uses SAM2 to generate pixel-precise masks
- Returns unified results with both 3D and 2D information

### SAMEnhancedBEVDetector
Enhances BEVNeXt detection using SAM2 features:
- Extracts SAM2 image encoder features
- Fuses with BEV features for improved detection
- Potentially increases detection accuracy

## Performance & Scalability

### Multi-GPU Performance
Expected performance improvements with multi-GPU training:

| Configuration | Training Speed | Memory Usage | Effective Batch Size |
|---------------|----------------|--------------|---------------------|
| Single GPU (2080Ti) | 1.0x baseline | 11GB | 4 |
| 2x GPU (DataParallel) | 1.6-1.8x | 22GB total | 8 |
| 2x GPU (DistributedDataParallel) | 1.8-1.95x | 22GB total | 8 |

### Optimizations
- **Automatic Mixed Precision**: Reduces memory usage by up to 50%
- **Gradient Accumulation**: Simulates larger batch sizes
- **Dynamic Batch Scaling**: Automatically adjusts batch size based on GPU count
- **NCCL Backend**: Optimized GPU-to-GPU communication
- **Distributed Data Loading**: Efficient data pipeline for multi-GPU setups

### Recommended Hardware
- **Minimum**: 1x GPU with 8GB+ VRAM
- **Recommended**: 2x GPU with 11GB+ VRAM each (tested on 2080Ti)
- **Optimal**: 4x GPU with 16GB+ VRAM each

## Documentation

### Additional Resources
- [MULTI_GPU_TRAINING.md](MULTI_GPU_TRAINING.md) - Comprehensive multi-GPU training guide
- [DOCKER.md](DOCKER.md) - Docker deployment and configuration
- [configs/](configs/) - Configuration files and examples
- [scripts/](scripts/) - Utility scripts and launchers

### Dataset Setup
For nuScenes dataset integration, see the training scripts:
- `setup_nuscenes_integration.py` - Automated nuScenes setup
- `nuscenes_loader.py` - Dataset loading utilities
- `scripts/check-dataset.sh` - Dataset validation

## Development

### File Structure

```
BEVNeXt-SAM2/
├── Dockerfile                              # Main Docker image
├── docker-compose.yml                      # Multi-service orchestration
├── MULTI_GPU_TRAINING.md                   # Multi-GPU training documentation
├── training/                               # Training scripts and utilities
│   ├── train_bevnext_sam2.py              # Main training script
│   ├── train_bevnext_sam2_nuscenes.py     # nuScenes training script
│   ├── multi_gpu_utils.py                 # Multi-GPU utilities
│   └── nuscenes_dataset_v2.py             # Enhanced dataset loader
├── scripts/                                # Docker and utility scripts
│   ├── build.sh                           # Docker build script
│   ├── run.sh                             # Main Docker interface
│   ├── launch_docker_multi_gpu.sh         # Docker multi-GPU launcher
│   ├── launch_distributed_training.sh     # Native multi-GPU launcher
│   ├── monitor_training.sh                # Training monitoring
│   └── setup_gpu.sh                       # GPU environment setup
├── bevnext/                               # BEVNeXt module
├── sam2_module/                           # SAM2 module
├── integration/                           # Integration code
├── validation/                            # Model evaluation and testing
├── data/                                  # Dataset storage
├── checkpoints/                           # Model checkpoints
├── configs/                               # Configuration files
│   ├── bevnext/                          # BEVNeXt configs
│   └── sam2/                             # SAM2 configs
└── tools/                                 # Training and inference utilities
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes:
   ```bash
   # Test basic functionality
   ./scripts/build.sh && ./scripts/run.sh demo
   
   # Test multi-GPU functionality (if available)
   python scripts/test_multi_gpu.py
   
   # Test single GPU compatibility
   python scripts/test_single_gpu.py
   ```
5. Submit a pull request

### Development Commands

```bash
# Run development environment
./scripts/run.sh dev

# Interactive shell for debugging
./scripts/run.sh shell

# Monitor training progress
./scripts/monitor_training.sh

# Check dataset integrity
./scripts/check-dataset.sh /path/to/dataset

# Test GPU setup
./scripts/setup_gpu.sh
```

## Citation

If you use this code in your research, please cite both original papers:

```bibtex
@inproceedings{li2024bevnext,
  title={BEVNeXt: Reviving Dense BEV Frameworks for 3D Object Detection},
  author={Li, Zhenxin and Lan, Shiyi and Alvarez, Jose M and Wu, Zuxuan},
  booktitle={CVPR},
  year={2024}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## License

This project inherits licenses from both original projects:
- BEVNeXt: Apache License 2.0
- SAM 2: Apache License 2.0

## Acknowledgments

This project builds upon:
- [BEVNeXt](https://github.com/woxihuanjiangguo/BEVNeXt)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 