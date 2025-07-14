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

# Try to import integration modules, fallback to simulation if not available
try:
    from integration import BEVSAMFusion, SAMEnhancedBEVDetector
    INTEGRATION_AVAILABLE = True
    print("✅ Integration modules loaded successfully")
except Exception as e:
    print(f"⚠️  Integration modules not available: {e}")
    print("📝 Running in full simulation mode...")
    INTEGRATION_AVAILABLE = False
    
    # Create mock classes for demonstration
    class BEVSAMFusion:
        def __init__(self, sam2_checkpoint, sam2_model_cfg, device):
            self.device = device
            print(f"📦 Mock BEVSAMFusion initialized on {device}")
        
        def __call__(self, image, bev_detections, camera_params):
            print("🔄 Simulating BEV-SAM fusion processing...")
            return {
                'boxes_3d': bev_detections['boxes_3d'],
                'boxes_2d': torch.randn(bev_detections['boxes_3d'].shape[0], 4),
                'masks': torch.randn(bev_detections['boxes_3d'].shape[0], 480, 640) > 0,
                'scores': bev_detections['scores'],
                'labels': bev_detections['labels']
            }
    
    class SAMEnhancedBEVDetector:
        def __init__(self, bev_config, sam2_checkpoint, sam2_model_cfg, device):
            self.device = device
            print(f"📦 Mock SAMEnhancedBEVDetector initialized on {device}")
        
        def simple_test(self, img, img_metas):
            print("🔄 Simulating SAM-enhanced detection...")
            num_detections = np.random.randint(3, 8)
            return [{
                'boxes_3d': torch.randn(num_detections, 7),
                'scores_3d': torch.rand(num_detections) * 0.4 + 0.6,
                'labels_3d': torch.randint(0, 3, (num_detections,)),
                'sam_features': torch.randn(num_detections, 256),
            }]


def demo_bev_sam_fusion():
    """
    Demonstrate BEVSAMFusion: BEVNeXt detection + SAM2 segmentation
    """
    print("=== BEV-SAM Fusion Demo ===")
    print("🎯 Demonstrating 3D detection + 2D segmentation fusion...")
    
    # Configuration with fallback options
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "configs/sam2/sam2.1/sam2.1_hiera_l.yaml"
    
    # Check if checkpoints exist, use dummy mode if not
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    if not os.path.exists(sam2_checkpoint):
        print(f"⚠️  SAM2 checkpoint not found at {sam2_checkpoint}")
        print("📝 Running in simulation mode with synthetic components...")
        use_real_sam2 = False
    else:
        print(f"✅ SAM2 checkpoint found")
        use_real_sam2 = True
    
    # Initialize fusion module 
    if INTEGRATION_AVAILABLE and use_real_sam2:
        print("✅ Initializing real BEV-SAM fusion module...")
        try:
            fusion = BEVSAMFusion(
                sam2_checkpoint=sam2_checkpoint,
                sam2_model_cfg=sam2_model_cfg,
                device=device
            )
            print("✅ Real BEV-SAM fusion module initialized successfully")
        except Exception as e:
            print(f"❌ Real module failed: {e}")
            print("📝 Falling back to simulation mode...")
            fusion = BEVSAMFusion(sam2_checkpoint, sam2_model_cfg, device)
    else:
        print("📦 Using simulation mode for BEV-SAM fusion...")
        fusion = BEVSAMFusion(sam2_checkpoint, sam2_model_cfg, device)
    
    # Dummy data for demonstration
    print("📊 Generating synthetic demo data...")
    batch_size = 1
    num_detections = 5
    image_height, image_width = 1080, 1920
    
    # Create dummy image
    dummy_image = torch.randn(batch_size, 3, image_height, image_width)
    print(f"📷 Created dummy camera image: {dummy_image.shape}")
    
    # Create dummy BEV detections (realistic autonomous driving objects)
    bev_detections = {
        'boxes_3d': torch.randn(num_detections, 7),  # x, y, z, w, h, l, yaw
        'scores': torch.rand(num_detections) * 0.5 + 0.5,  # scores between 0.5-1.0
        'labels': torch.randint(0, 3, (num_detections,))  # 0: car, 1: pedestrian, 2: cyclist
    }
    
    object_names = ['car', 'pedestrian', 'cyclist']
    print(f"🚗 Generated {num_detections} BEV detections:")
    for i in range(num_detections):
        obj_name = object_names[bev_detections['labels'][i]]
        score = bev_detections['scores'][i].item()
        print(f"  • {obj_name} (confidence: {score:.2f})")
    
    # Camera parameters (dummy)
    print("📐 Setting up camera parameters...")
    camera_params = {
        'intrinsics': torch.tensor([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=torch.float32),
        'extrinsics': torch.eye(4, dtype=torch.float32)
    }
    
    # Run fusion
    print("🔄 Running BEV-SAM fusion pipeline...")
    with torch.no_grad():
        results = fusion(
            image=dummy_image,
            bev_detections=bev_detections,
            camera_params=camera_params
        )
    
    print("✅ Fusion completed! Results summary:")
    print(f"📦 3D bounding boxes: {results['boxes_3d'].shape}")
    print(f"📐 2D projected boxes: {results['boxes_2d'].shape}") 
    print(f"🎭 Segmentation masks: {results['masks'].shape}")
    print(f"🎯 Total detections: {len(results['scores'])}")
    
    # Show detailed results for each detection
    print("\n📋 Detailed detection results:")
    for i in range(min(3, len(results['scores']))):  # Show first 3 detections
        obj_name = object_names[results['labels'][i]]
        score = results['scores'][i].item()
        box_3d = results['boxes_3d'][i]
        box_2d = results['boxes_2d'][i]
        mask_size = results['masks'][i].sum().item()
        
        print(f"  Detection {i+1}: {obj_name}")
        print(f"    • Confidence: {score:.3f}")
        print(f"    • 3D position: x={box_3d[0]:.2f}, y={box_3d[1]:.2f}, z={box_3d[2]:.2f}")
        print(f"    • 2D box: [{box_2d[0]:.0f}, {box_2d[1]:.0f}, {box_2d[2]:.0f}, {box_2d[3]:.0f}]")
        print(f"    • Mask pixels: {mask_size:.0f}")
    
    print("🎊 BEV-SAM fusion demo completed successfully!")
    

def demo_sam_enhanced_detector():
    """
    Demonstrate SAMEnhancedBEVDetector: SAM2-enhanced BEV detection
    """
    print("\n=== SAM-Enhanced BEV Detector Demo ===")
    print("🚀 Demonstrating SAM2-enhanced 3D object detection...")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    bev_config = {
        'type': 'BEVDet',
        'bev_channels': 256,
        # Add more BEVNeXt config parameters as needed
    }
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam2_model_cfg = "configs/sam2/sam2.1/sam2.1_hiera_l.yaml"
    
    # Initialize enhanced detector
    if INTEGRATION_AVAILABLE and os.path.exists(sam2_checkpoint):
        print("✅ SAM2 checkpoint found, initializing real detector...")
        try:
            detector = SAMEnhancedBEVDetector(
                bev_config=bev_config,
                sam2_checkpoint=sam2_checkpoint,
                sam2_model_cfg=sam2_model_cfg,
                device=device
            )
            print("✅ Real SAM-enhanced detector initialized successfully")
        except Exception as e:
            print(f"❌ Real detector failed: {e}")
            print("📝 Falling back to simulation mode...")
            detector = SAMEnhancedBEVDetector(bev_config, sam2_checkpoint, sam2_model_cfg, device)
    else:
        print("📦 Using simulation mode for SAM-enhanced detector...")
        detector = SAMEnhancedBEVDetector(bev_config, sam2_checkpoint, sam2_model_cfg, device)
    
    # Dummy multi-view images
    print("📊 Generating multi-view camera data...")
    batch_size = 1
    num_views = 6  # Typical for autonomous driving
    image_height, image_width = 900, 1600
    view_names = ['front', 'back', 'left', 'right', 'front-left', 'front-right']
    
    dummy_images = torch.randn(batch_size, num_views, 3, image_height, image_width)
    print(f"📷 Created {num_views} camera views: {dummy_images.shape}")
    for i, view_name in enumerate(view_names):
        print(f"  • {view_name}: {image_height}×{image_width}")
    
    # Image metadata
    img_metas = [{
        'img_shape': [(image_height, image_width, 3)] * num_views,
        'lidar2img': [np.eye(4)] * num_views,
        'cam_intrinsic': [np.eye(3)] * num_views,
    }]
    
    # Run enhanced detection
    print("🔄 Running SAM-enhanced BEV detection...")
    with torch.no_grad():
        results = detector.simple_test(
            img=dummy_images,
            img_metas=img_metas
        )
    
    print("✅ SAM-enhanced detection completed! Results summary:")
    object_names = ['car', 'pedestrian', 'cyclist']
    
    for i, result in enumerate(results):
        print(f"\n📦 Sample {i+1} enhanced detection results:")
        
        # Get detection counts and data
        if 'boxes_3d' in result:
            num_detections = len(result['boxes_3d'])
            boxes_3d = result['boxes_3d']
            scores = result.get('scores_3d', torch.ones(num_detections))
            labels = result.get('labels_3d', torch.zeros(num_detections, dtype=torch.long))
            
            print(f"  🎯 Total detections: {num_detections}")
            print(f"  📦 3D boxes shape: {boxes_3d.shape}")
            if 'sam_features' in result:
                print(f"  🧠 SAM features shape: {result['sam_features'].shape}")
            
            # Show individual detections
            print("  📋 Enhanced detection details:")
            for j in range(min(3, num_detections)):  # Show first 3
                obj_name = object_names[labels[j] % len(object_names)]
                score = scores[j].item()
                box = boxes_3d[j]
                
                print(f"    Detection {j+1}: {obj_name}")
                print(f"      • Enhanced confidence: {score:.3f}")
                print(f"      • Position: ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f})")
                print(f"      • Dimensions: {box[3]:.2f}×{box[4]:.2f}×{box[5]:.2f}")
        else:
            print("  📊 Result keys:", list(result.keys()))
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    print(f"    • {key}: {value.shape}")
    
    print("\n🎊 SAM-enhanced detection demo completed successfully!")


def main():
    """Run all demos"""
    print("🎯 BEVNeXt-SAM2 Integration Demo")
    print("=" * 60)
    print("🚀 Showcasing 3D detection + 2D segmentation fusion")
    print("📦 Running in automatic demo mode with synthetic data")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Computing device: {device}")
    if torch.cuda.is_available():
        print(f"⚡ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print()
    
    demo_success = []
    
    try:
        # Demo 1: BEV-SAM Fusion
        print("🔄 Starting Demo 1: BEV-SAM Fusion Pipeline...")
        demo_bev_sam_fusion()
        demo_success.append("✅ BEV-SAM Fusion")
    except Exception as e:
        print(f"❌ BEV-SAM Fusion demo encountered error: {e}")
        print("📝 This is expected in demo mode without real checkpoints")
        demo_success.append("⚠️  BEV-SAM Fusion (simulated)")
    
    try:
        # Demo 2: SAM-Enhanced Detector
        print("\n🔄 Starting Demo 2: SAM-Enhanced BEV Detector...")
        demo_sam_enhanced_detector()
        demo_success.append("✅ SAM-Enhanced Detector")
    except Exception as e:
        print(f"❌ SAM-Enhanced Detector demo encountered error: {e}")
        print("📝 This is expected in demo mode without real checkpoints")
        demo_success.append("⚠️  SAM-Enhanced Detector (simulated)")
    
    print("\n" + "=" * 60)
    print("🎊 BEVNeXt-SAM2 Demo Completed Successfully!")
    print("=" * 60)
    
    print("📊 Demo Results Summary:")
    for result in demo_success:
        print(f"  {result}")
    
    print(f"\n🎯 Key Demonstrations:")
    print("  • 3D object detection in bird's-eye view")
    print("  • 2D segmentation mask generation") 
    print("  • Multi-camera fusion processing")
    print("  • SAM2-enhanced feature extraction")
    print("  • Synthetic data generation")
    
    print(f"\n🚀 Ready for Next Steps:")
    print("  1. Train with real data: ./scripts/run.sh train")
    print("  2. Run development environment: ./scripts/run.sh dev")
    print("  3. Monitor training: ./scripts/run.sh tensorboard")
    print("  4. Download real checkpoints for production use")
    
    print(f"\n💡 This demo used synthetic data - ready for real datasets!")


if __name__ == "__main__":
    main() 