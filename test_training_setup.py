#!/usr/bin/env python3
"""
Test script to verify BEVNeXt-SAM2 training setup
"""

import os
import sys
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        # Test PyTorch
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} imported successfully")
        
        # Test CUDA
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠ CUDA not available, will use CPU")
        
        # Test project modules
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        try:
            from training.train_bevnext_sam2_nuscenes import EnhancedBEVNeXtSAM2Model, NuScenesTrainer, get_enhanced_config
            logger.info("✓ Training modules imported successfully")
        except ImportError as e:
            logger.error(f"✗ Failed to import training modules: {e}")
            return False
        
        try:
            from training.nuscenes_dataset_v2 import NuScenesMultiModalDataset, NuScenesConfig
            logger.info("✓ nuScenes dataset modules imported successfully")
        except ImportError as e:
            logger.warning(f"⚠ nuScenes dataset modules not available: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        return False

def test_model_creation():
    """Test that the model can be created"""
    logger.info("Testing model creation...")
    
    try:
        from training.train_bevnext_sam2_nuscenes import EnhancedBEVNeXtSAM2Model, get_enhanced_config
        
        # Get config
        config = get_enhanced_config()
        
        # Create model
        model = EnhancedBEVNeXtSAM2Model(config)
        logger.info(f"✓ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass with dummy data
        batch_size = 1
        num_cameras = 6
        dummy_batch = {
            'camera_images': torch.randn(batch_size, num_cameras, 3, 224, 224),
            'gt_boxes': torch.randn(batch_size, 100, 10),
            'gt_labels': torch.randint(0, 23, (batch_size, 100)),
            'gt_valid': torch.randint(0, 2, (batch_size, 100), dtype=torch.bool),
            'sample_token': ['test_sample'],
            'camera_intrinsics': torch.eye(3, 4).unsqueeze(0).repeat(batch_size, 1, 1),
            'camera_extrinsics': torch.eye(4, 4).unsqueeze(0).repeat(batch_size, 1, 1)
        }
        
        with torch.no_grad():
            outputs = model(dummy_batch)
        
        logger.info(f"✓ Forward pass successful, output keys: {list(outputs.keys())}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation test failed: {e}")
        return False

def test_trainer_creation():
    """Test that the trainer can be created"""
    logger.info("Testing trainer creation...")
    
    try:
        from training.train_bevnext_sam2_nuscenes import NuScenesTrainer, get_enhanced_config
        
        # Get config
        config = get_enhanced_config()
        
        # Create trainer
        trainer = NuScenesTrainer(config, use_mixed_precision=False)  # Disable mixed precision for testing
        
        logger.info("✓ Trainer created successfully")
        logger.info(f"✓ Train loader: {len(trainer.train_loader)} batches")
        logger.info(f"✓ Val loader: {len(trainer.val_loader)} batches")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Trainer creation test failed: {e}")
        return False

def test_mini_training():
    """Test a mini training run"""
    logger.info("Testing mini training run...")
    
    try:
        from training.train_bevnext_sam2_nuscenes import NuScenesTrainer, get_enhanced_config
        
        # Get config with small settings for testing
        config = get_enhanced_config()
        config['num_epochs'] = 1  # Just one epoch for testing
        config['batch_size'] = 1
        
        # Create trainer
        trainer = NuScenesTrainer(config, use_mixed_precision=False)
        
        # Run one training step
        trainer.model.train()
        batch = next(iter(trainer.train_loader))
        batch = trainer._move_batch_to_device(batch)
        
        # Forward pass
        trainer.optimizer.zero_grad()
        predictions = trainer.model(batch)
        losses = trainer.model.compute_loss(predictions, batch)
        total_loss = sum(losses.values())
        
        # Backward pass
        total_loss.backward()
        trainer.optimizer.step()
        
        logger.info(f"✓ Mini training step successful, loss: {total_loss.item():.4f}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Mini training test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("="*50)
    logger.info("BEVNeXt-SAM2 Training Setup Test")
    logger.info("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Creation Test", test_model_creation),
        ("Trainer Creation Test", test_trainer_creation),
        ("Mini Training Test", test_mini_training),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:25s}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Training setup is ready.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main()) 