#!/usr/bin/env python3
"""
Test script for single-GPU training implementation
Tests the multi-GPU code but with single GPU to verify the implementation is correct
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.multi_gpu_utils import (
    MultiGPUWrapper,
    setup_multi_gpu_training,
    log_gpu_memory_usage
)

def test_single_gpu_detection():
    """Test GPU detection and configuration for single GPU"""
    print("=== Single GPU Detection Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA available with {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    return True

def test_single_gpu_wrapper():
    """Test MultiGPUWrapper functionality with single GPU"""
    print("\n=== Single GPU Wrapper Test ===")
    
    try:
        # Test single GPU mode
        wrapper = MultiGPUWrapper(gpu_ids="0", distributed=False)
        print(f"✅ Single GPU wrapper created: {wrapper.gpu_ids}")
        print(f"   Device: {wrapper.device}")
        print(f"   Num GPUs: {wrapper.num_gpus}")
        print(f"   Is main process: {wrapper.is_main_process()}")
        
        return True
    except Exception as e:
        print(f"❌ Single GPU wrapper test failed: {e}")
        return False

def test_model_wrapping_single_gpu():
    """Test model wrapping with single GPU (should work without parallelization)"""
    print("\n=== Single GPU Model Wrapping Test ===")
    
    try:
        # Create a simple test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(5, 1)
                
            def forward(self, x):
                return self.linear2(torch.relu(self.linear1(x)))
        
        # Test single GPU wrapping (no actual parallelization)
        model = TestModel()
        wrapper = MultiGPUWrapper(gpu_ids="0", distributed=False)
        wrapped_model = wrapper.wrap_model(model)
        
        print(f"✅ Model wrapped for single GPU")
        print(f"   Model type: {type(wrapped_model)}")
        print(f"   Model device: {next(wrapped_model.parameters()).device}")
        
        # Test forward pass
        test_input = torch.randn(4, 10).to(wrapper.device)
        output = wrapped_model(test_input)
        print(f"✅ Forward pass successful: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model wrapping test failed: {e}")
        return False

def test_learning_rate_scaling_single_gpu():
    """Test learning rate scaling for single GPU (should return original LR)"""
    print("\n=== Single GPU Learning Rate Scaling Test ===")
    
    try:
        wrapper = MultiGPUWrapper(gpu_ids="0", distributed=False)
        
        base_lr = 1e-4
        linear_lr = wrapper.scale_learning_rate(base_lr, 'linear')
        sqrt_lr = wrapper.scale_learning_rate(base_lr, 'sqrt')
        none_lr = wrapper.scale_learning_rate(base_lr, 'none')
        
        print(f"✅ Learning rate scaling test (single GPU):")
        print(f"   Base LR: {base_lr}")
        print(f"   Linear scaling: {linear_lr} (expected: {base_lr})")
        print(f"   Sqrt scaling: {sqrt_lr} (expected: {base_lr})")
        print(f"   No scaling: {none_lr} (expected: {base_lr})")
        
        # For single GPU, all should return base_lr
        assert abs(linear_lr - base_lr) < 1e-6, "Single GPU linear scaling should return base LR"
        assert abs(sqrt_lr - base_lr) < 1e-6, "Single GPU sqrt scaling should return base LR"
        assert abs(none_lr - base_lr) < 1e-6, "Single GPU no scaling should return base LR"
        
        return True
    except Exception as e:
        print(f"❌ Learning rate scaling test failed: {e}")
        return False

def test_training_args_compatibility():
    """Test that our training scripts can handle multi-GPU arguments even with single GPU"""
    print("\n=== Training Arguments Compatibility Test ===")
    
    try:
        # Test if we can import the training modules
        from training.train_bevnext_sam2 import BEVNeXtSAM2Model, Trainer
        print("✅ Successfully imported train_bevnext_sam2 module")
        
        # Test basic config
        config = {
            'd_model': 256,
            'num_transformer_layers': 6,
            'image_size': [224, 224],
            'bev_size': [200, 200],
            'num_queries': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 1,
            'batch_size': 2,
            'num_workers': 2,
            'output_dir': 'outputs/test'
        }
        
        # Test creating trainer with multi-GPU parameters
        trainer = Trainer(
            config=config,
            use_mixed_precision=True,
            gpu_ids="0",  # Single GPU
            distributed=False,
            gradient_accumulation=1,
            lr_scaling='linear'
        )
        
        print("✅ Successfully created Trainer with multi-GPU parameters")
        print(f"   Device: {trainer.device}")
        print(f"   GPU count: {trainer.gpu_wrapper.num_gpus}")
        print(f"   Mixed precision: {trainer.use_mixed_precision}")
        
        return True
    except Exception as e:
        print(f"❌ Training arguments compatibility test failed: {e}")
        return False

def test_memory_monitoring_single_gpu():
    """Test GPU memory monitoring with single GPU"""
    print("\n=== Single GPU Memory Monitoring Test ===")
    
    try:
        wrapper = MultiGPUWrapper(gpu_ids="0", distributed=False)
        
        # Allocate some GPU memory
        dummy_tensor = torch.randn(1000, 1000).to(wrapper.device)
        
        stats = wrapper.get_gpu_memory_stats()
        print(f"✅ Memory stats retrieved:")
        for gpu_name, gpu_stats in stats.items():
            print(f"   {gpu_name}: {gpu_stats['allocated']:.2f}GB allocated")
        
        # Test memory logging
        log_gpu_memory_usage(wrapper, "Test: ")
        
        # Clean up
        del dummy_tensor
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")
        return False

def main():
    """Run all tests for single GPU compatibility"""
    print("BEVNeXt-SAM2 Single GPU Compatibility Test")
    print("(Testing multi-GPU code with single GPU)")
    print("=" * 50)
    
    tests = [
        ("Single GPU Detection", test_single_gpu_detection),
        ("Single GPU Wrapper", test_single_gpu_wrapper),
        ("Single GPU Model Wrapping", test_model_wrapping_single_gpu),
        ("Single GPU LR Scaling", test_learning_rate_scaling_single_gpu),
        ("Training Args Compatibility", test_training_args_compatibility),
        ("Single GPU Memory Monitoring", test_memory_monitoring_single_gpu),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-GPU implementation is compatible with single GPU.")
        print("💡 The code is ready to run on your server with 2x 2080Ti GPUs.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())