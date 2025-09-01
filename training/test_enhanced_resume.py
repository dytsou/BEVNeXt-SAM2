#!/usr/bin/env python3
"""
Test Script for Enhanced Resume Functionality

This script tests the enhanced resume system to ensure it correctly
handles checkpoint loading and training state restoration.

Usage:
    python training/test_enhanced_resume.py
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.enhanced_resume import EnhancedResumeManager, create_enhanced_resume_manager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple model for testing resume functionality"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 50, output_size: int = 1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


def create_test_config() -> Dict:
    """Create test configuration"""
    return {
        'd_model': 50,
        'num_classes': 1,
        'num_queries': 10,
        'bev_size': [32, 32],
        'batch_size': 2,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 10
    }


def test_basic_resume():
    """Test basic checkpoint save and resume functionality"""
    logger.info("🧪 Testing basic resume functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model and training components
        model = SimpleTestModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        config = create_test_config()
        
        # Create resume manager
        resume_manager = create_enhanced_resume_manager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=temp_path,
            scaler=scaler,
            config=config
        )
        
        # Set some training state
        resume_manager.epoch = 5
        resume_manager.best_val_loss = 0.123
        resume_manager.batch_step = 100
        resume_manager.training_stats = defaultdict(list)
        resume_manager.training_stats['train_loss'] = [1.0, 0.8, 0.6, 0.4, 0.2]
        resume_manager.training_stats['val_loss'] = [1.1, 0.9, 0.7, 0.5, 0.3]
        
        # Create and save checkpoint
        logger.info("📁 Creating enhanced checkpoint...")
        checkpoint = resume_manager.create_enhanced_checkpoint()
        checkpoint_path = temp_path / 'test_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
        logger.info(f"   📊 Epoch: {checkpoint['epoch']}")
        logger.info(f"   📈 Best val loss: {checkpoint['best_val_loss']}")
        logger.info(f"   🔢 Batch step: {checkpoint['batch_step']}")
        
        # Create new model and components for loading
        new_model = SimpleTestModel().to(device)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.1)
        new_scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Create new resume manager
        new_resume_manager = create_enhanced_resume_manager(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=device,
            output_dir=temp_path,
            scaler=new_scaler,
            config=config
        )
        
        # Test resume
        logger.info("🔄 Testing resume from checkpoint...")
        result = new_resume_manager.resume_from_checkpoint(checkpoint_path)
        
        if result.success:
            logger.info("✅ Resume test PASSED!")
            logger.info(f"   📊 Restored epoch: {result.restored_epoch}")
            logger.info(f"   📈 Restored step: {result.restored_step}")
            logger.info(f"   📉 Best val loss: {new_resume_manager.best_val_loss}")
            
            # Verify state restoration
            assert new_resume_manager.epoch == 5, f"Epoch mismatch: {new_resume_manager.epoch} != 5"
            assert abs(new_resume_manager.best_val_loss - 0.123) < 1e-6, f"Best val loss mismatch"
            assert new_resume_manager.batch_step == 100, f"Batch step mismatch: {new_resume_manager.batch_step} != 100"
            
            if result.warnings:
                logger.warning("   ⚠️  Warnings during resume:")
                for warning in result.warnings:
                    logger.warning(f"      - {warning}")
        else:
            logger.error("❌ Resume test FAILED!")
            logger.error(f"   Error: {result.message}")
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"   Warning: {warning}")
            return False
    
    return True


def test_auto_resume():
    """Test auto-resume functionality"""
    logger.info("🧪 Testing auto-resume functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = create_test_config()
        
        # Create multiple checkpoints at different epochs
        checkpoints_info = [
            {'epoch': 3, 'val_loss': 0.8, 'name': 'checkpoint_epoch_3.pth'},
            {'epoch': 7, 'val_loss': 0.4, 'name': 'checkpoint_epoch_7.pth'},
            {'epoch': 5, 'val_loss': 0.6, 'name': 'checkpoint_epoch_5.pth'},
            {'epoch': 10, 'val_loss': 0.2, 'name': 'checkpoint_latest.pth'},  # Should be selected
        ]
        
        for ckpt_info in checkpoints_info:
            # Create model components
            model = SimpleTestModel().to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            # Create resume manager and set state
            resume_manager = create_enhanced_resume_manager(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                output_dir=temp_path,
                config=config
            )
            
            resume_manager.epoch = ckpt_info['epoch']
            resume_manager.best_val_loss = ckpt_info['val_loss']
            resume_manager.batch_step = ckpt_info['epoch'] * 50  # Simulate progress
            
            # Save checkpoint
            checkpoint = resume_manager.create_enhanced_checkpoint()
            checkpoint_path = temp_path / ckpt_info['name']
            torch.save(checkpoint, checkpoint_path)
            
            logger.info(f"   📁 Created: {ckpt_info['name']} (epoch {ckpt_info['epoch']})")
        
        # Test auto-resume (should pick the latest/best checkpoint)
        new_model = SimpleTestModel().to(device)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.1)
        
        auto_resume_manager = create_enhanced_resume_manager(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=device,
            output_dir=temp_path,
            config=config
        )
        
        logger.info("🔍 Testing auto-resume...")
        result = auto_resume_manager.auto_resume()
        
        if result.success:
            logger.info("✅ Auto-resume test PASSED!")
            logger.info(f"   📁 Selected: {result.checkpoint_path.name}")
            logger.info(f"   📊 Restored epoch: {result.restored_epoch}")
            logger.info(f"   📈 Best val loss: {auto_resume_manager.best_val_loss}")
            
            # Should have selected the latest checkpoint (epoch 10)
            assert auto_resume_manager.epoch == 10, f"Expected epoch 10, got {auto_resume_manager.epoch}"
            
            if result.warnings:
                logger.warning("   ⚠️  Warnings during auto-resume:")
                for warning in result.warnings:
                    logger.warning(f"      - {warning}")
        else:
            logger.error("❌ Auto-resume test FAILED!")
            logger.error(f"   Error: {result.message}")
            return False
    
    return True


def test_error_handling():
    """Test error handling for corrupt/invalid checkpoints"""
    logger.info("🧪 Testing error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = create_test_config()
        
        # Create model components
        model = SimpleTestModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        resume_manager = create_enhanced_resume_manager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=temp_path,
            config=config
        )
        
        # Test 1: Non-existent checkpoint
        logger.info("   🔍 Testing non-existent checkpoint...")
        result = resume_manager.resume_from_checkpoint(temp_path / 'nonexistent.pth')
        assert not result.success, "Should fail for non-existent checkpoint"
        logger.info("   ✅ Non-existent checkpoint handled correctly")
        
        # Test 2: Corrupt checkpoint (empty file)
        logger.info("   🔍 Testing corrupt checkpoint...")
        corrupt_path = temp_path / 'corrupt.pth'
        corrupt_path.write_text("This is not a valid checkpoint")
        
        result = resume_manager.resume_from_checkpoint(corrupt_path)
        assert not result.success, "Should fail for corrupt checkpoint"
        logger.info("   ✅ Corrupt checkpoint handled correctly")
        
        # Test 3: Checkpoint with missing required keys
        logger.info("   🔍 Testing incomplete checkpoint...")
        incomplete_checkpoint = {'epoch': 5}  # Missing required keys
        incomplete_path = temp_path / 'incomplete.pth'
        torch.save(incomplete_checkpoint, incomplete_path)
        
        result = resume_manager.resume_from_checkpoint(incomplete_path)
        assert not result.success, "Should fail for incomplete checkpoint"
        logger.info("   ✅ Incomplete checkpoint handled correctly")
        
        logger.info("✅ Error handling tests PASSED!")
    
    return True


def test_config_compatibility():
    """Test configuration compatibility checking"""
    logger.info("🧪 Testing configuration compatibility...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create checkpoint with specific config
        original_config = create_test_config()
        original_config['d_model'] = 64
        original_config['num_classes'] = 5
        
        model = SimpleTestModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        resume_manager = create_enhanced_resume_manager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=temp_path,
            config=original_config
        )
        
        # Save checkpoint
        checkpoint = resume_manager.create_enhanced_checkpoint()
        checkpoint_path = temp_path / 'config_test.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Test with different config (should generate warnings or fail based on strict_config_check)
        different_config = create_test_config()
        different_config['d_model'] = 128  # Different value
        different_config['num_classes'] = 10  # Different value
        
        new_model = SimpleTestModel().to(device)
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.1)
        
        # Test with strict config checking (should fail)
        logger.info("   🔍 Testing strict config checking...")
        strict_resume_manager = create_enhanced_resume_manager(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=device,
            output_dir=temp_path,
            config=different_config,
            strict_config_check=True
        )
        
        result = strict_resume_manager.resume_from_checkpoint(checkpoint_path)
        assert not result.success, "Should fail with strict config checking"
        logger.info("   ✅ Strict config checking works correctly")
        
        # Test with lenient config checking (should succeed with warnings)
        logger.info("   🔍 Testing lenient config checking...")
        lenient_resume_manager = create_enhanced_resume_manager(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device=device,
            output_dir=temp_path,
            config=different_config,
            strict_config_check=False
        )
        
        result = lenient_resume_manager.resume_from_checkpoint(checkpoint_path)
        assert result.success, "Should succeed with lenient config checking"
        assert len(result.warnings) > 0, "Should have warnings about config mismatches"
        logger.info("   ✅ Lenient config checking works correctly")
        
        logger.info("✅ Configuration compatibility tests PASSED!")
    
    return True


def run_all_tests():
    """Run all enhanced resume tests"""
    logger.info("🚀 Starting Enhanced Resume System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Resume", test_basic_resume),
        ("Auto Resume", test_auto_resume),
        ("Error Handling", test_error_handling),
        ("Config Compatibility", test_config_compatibility),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test...")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                logger.info(f"✅ {test_name} Test: PASSED")
            else:
                logger.error(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} Test: FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! Enhanced resume system is working correctly.")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) FAILED! Please check the enhanced resume system.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
