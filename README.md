# BEVNeXt-SAM2: Unified 3D Object Detection and Segmentation

This repository merges [BEVNeXt](https://github.com/woxihuanjiangguo/BEVNeXt) and [SAM 2](https://github.com/facebookresearch/sam2) to create a unified framework for 3D object detection with enhanced segmentation capabilities.

## Overview

BEVNeXt-SAM2 combines:
- **BEVNeXt**: State-of-the-art Bird's Eye View (BEV) 3D object detection
- **SAM 2**: Segment Anything Model for high-quality image and video segmentation

The integration enables:
- 3D object detection from multi-view cameras using BEV representation
- Instance segmentation masks for detected objects using SAM 2
- Enhanced perception by combining geometric (3D boxes) and appearance (segmentation) information

## Architecture

```
BEVNeXt-SAM2/
├── bevnext/          # BEVNeXt 3D detection module
├── sam2_module/      # SAM 2 segmentation module  
├── integration/      # Fusion modules combining both
├── configs/          # Configuration files
│   ├── bevnext/      # BEVNeXt configs
│   └── sam2/         # SAM 2 configs
├── tools/            # Training and inference scripts
├── examples/         # Demo scripts
├── scripts/          # Docker and utility scripts
└── requirements/     # Dependencies
```

## Installation

### Option 1: Docker (Recommended)

Docker provides the easiest way to get started with all dependencies pre-installed:

#### Quick Start
```bash
# Build the Docker image
./scripts/build.sh

# Run the demo
./scripts/run.sh demo

# Start development environment with Jupyter
./scripts/run.sh dev
```

#### Using Docker Compose
```bash
# Run demo
docker-compose up --build bevnext-sam2

# Development with Jupyter Lab
docker-compose up --build dev
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

Training scripts for both components are provided:

```bash
# Native installation
python tools/train_bevnext.py --config configs/bevnext/bevnext-stage1.py
python tools/train_sam2.py --config configs/sam2/sam2_finetune.yaml

# Docker
./scripts/run.sh train
docker-compose exec train bash
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
- **train**: Training environment
- **inference**: Inference environment
- **tensorboard**: TensorBoard monitoring

### Examples

```bash
# Quick demo
./scripts/run.sh demo

# Development with Jupyter (http://localhost:8888)
./scripts/run.sh dev

# Interactive shell
./scripts/run.sh shell

# Training on specific GPU
./scripts/run.sh train --gpu 1

# TensorBoard monitoring
./scripts/run.sh tensorboard
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

### 4. Flexible Architecture
Modular design allows using components independently or together.

### 5. Containerized Deployment
Full Docker support with multiple deployment options.

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

## Development

### File Structure

```
BEVNeXt-SAM2/
├── Dockerfile              # Main Docker image
├── docker-compose.yml      # Multi-service orchestration
├── .dockerignore           # Docker build exclusions
├── DOCKER.md               # Docker documentation
├── scripts/
│   ├── build.sh            # Docker build script
│   └── run.sh              # Docker run script
├── bevnext/                # BEVNeXt module
├── sam2_module/            # SAM2 module
├── integration/            # Integration code
├── examples/               # Demo scripts
└── configs/                # Configuration files
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker: `./scripts/build.sh && ./scripts/run.sh demo`
5. Submit a pull request

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