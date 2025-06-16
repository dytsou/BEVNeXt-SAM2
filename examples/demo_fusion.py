"""
Demo script for BEVNeXt-SAM2 fusion

This script demonstrates how to use the merged BEVNeXt and SAM2 models
for enhanced 3D object detection with instance segmentation.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration import BEVSAMFusion, SAMEnhancedBEVDetector


def demo_bev_sam_fusion():
    """
    Demonstrate BEVSAMFusion: BEVNeXt detection + SAM2 segmentation
    """
    print("=== BEV-SAM Fusion Demo ===")
    
    # Configuration
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "configs/sam2/sam2.1/sam2.1_hiera_l.yaml"
    
    # Initialize fusion module
    fusion = BEVSAMFusion(
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_cfg=sam2_model_cfg,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Dummy data for demonstration
    batch_size = 1
    num_detections = 5
    image_height, image_width = 1080, 1920
    
    # Create dummy image
    dummy_image = torch.randn(batch_size, 3, image_height, image_width)
    
    # Create dummy BEV detections
    bev_detections = {
        'boxes_3d': torch.randn(num_detections, 7),  # x, y, z, w, h, l, yaw
        'scores': torch.rand(num_detections),
        'labels': torch.randint(0, 10, (num_detections,))
    }
    
    # Camera parameters (dummy)
    camera_params = {
        'intrinsics': torch.tensor([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=torch.float32),
        'extrinsics': torch.eye(4, dtype=torch.float32)
    }
    
    # Run fusion
    with torch.no_grad():
        results = fusion(
            image=dummy_image,
            bev_detections=bev_detections,
            camera_params=camera_params
        )
    
    print(f"Results:")
    print(f"- 3D boxes shape: {results['boxes_3d'].shape}")
    print(f"- 2D boxes shape: {results['boxes_2d'].shape}")
    print(f"- Masks shape: {results['masks'].shape}")
    print(f"- Number of detections: {len(results['scores'])}")
    

def demo_sam_enhanced_detector():
    """
    Demonstrate SAMEnhancedBEVDetector: SAM2-enhanced BEV detection
    """
    print("\n=== SAM-Enhanced BEV Detector Demo ===")
    
    # Configuration
    bev_config = {
        'type': 'BEVDet',
        'bev_channels': 256,
        # Add more BEVNeXt config parameters as needed
    }
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "configs/sam2/sam2.1/sam2.1_hiera_l.yaml"
    
    # Initialize enhanced detector
    detector = SAMEnhancedBEVDetector(
        bev_config=bev_config,
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_cfg=sam2_model_cfg,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Dummy multi-view images
    batch_size = 1
    num_views = 6  # Typical for autonomous driving (front, back, left, right, front-left, front-right)
    image_height, image_width = 900, 1600
    
    dummy_images = torch.randn(batch_size, num_views, 3, image_height, image_width)
    
    # Image metadata
    img_metas = [{
        'img_shape': [(image_height, image_width, 3)] * num_views,
        'lidar2img': [np.eye(4)] * num_views,
        'cam_intrinsic': [np.eye(3)] * num_views,
    }]
    
    # Run enhanced detection
    with torch.no_grad():
        results = detector.simple_test(
            img=dummy_images,
            img_metas=img_metas
        )
    
    print(f"Enhanced detection results for {len(results)} samples")
    for i, result in enumerate(results):
        print(f"Sample {i}:")
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape {value.shape}")
            else:
                print(f"  - {key}: {type(value)}")


def main():
    """Run all demos"""
    print("BEVNeXt-SAM2 Integration Demo")
    print("=" * 50)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Demo 1: BEV-SAM Fusion
        demo_bev_sam_fusion()
    except Exception as e:
        print(f"BEV-SAM Fusion demo failed: {e}")
        print("Note: Make sure SAM2 checkpoints are downloaded")
    
    try:
        # Demo 2: SAM-Enhanced Detector
        demo_sam_enhanced_detector()
    except Exception as e:
        print(f"SAM-Enhanced Detector demo failed: {e}")
        print("Note: Some components may need additional setup")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Download model checkpoints")
    print("2. Prepare your dataset")
    print("3. Modify configs for your use case")
    print("4. Run training or inference")


if __name__ == "__main__":
    main() 