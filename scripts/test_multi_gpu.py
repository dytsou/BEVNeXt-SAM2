#!/usr/bin/env python3
"""
Test script for multi-GPU training implementation
Tests both DataParallel and DistributedDataParallel modes
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

def test_gpu_detection():
    """Test GPU detection and configuration"""
    print("=== GPU Detection Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA available with {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    return num_gpus >= 2

def test_multi_gpu_wrapper():
    """Test MultiGPUWrapper functionality"""
    print("\n=== MultiGPUWrapper Test ===")
    
    try:
        # Test DataParallel mode
        wrapper_dp = MultiGPUWrapper(gpu_ids="0,1", distributed=False)
        print(f"✅ DataParallel wrapper created: {wrapper_dp.gpu_ids}")
        print(f"   Device: {wrapper_dp.device}")
        print(f"   Num GPUs: {wrapper_dp.num_gpus}")
        
        # Test DistributedDataParallel mode (setup only)
        wrapper_ddp = MultiGPUWrapper(gpu_ids="0,1", distributed=True)
        print(f"✅ DistributedDataParallel wrapper created: {wrapper_ddp.gpu_ids}")
        print(f"   Device: {wrapper_ddp.device}")
        print(f"   Num GPUs: {wrapper_ddp.num_gpus}")
        
        return True
    except Exception as e:
        print(f"❌ MultiGPUWrapper test failed: {e}")
        return False

def test_model_wrapping():
    """Test model wrapping with different parallel modes"""
    print("\n=== Model Wrapping Test ===")
    
    try:
        # Create a simple test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(5, 1)
                
            def forward(self, x):
                return self.linear2(torch.relu(self.linear1(x)))
        
        # Test DataParallel wrapping
        model = TestModel()
        wrapper = MultiGPUWrapper(gpu_ids="0,1", distributed=False)
        wrapped_model = wrapper.wrap_model(model)
        
        print(f"✅ Model wrapped with DataParallel")
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

def test_learning_rate_scaling():
    """Test learning rate scaling for multi-GPU"""
    print("\n=== Learning Rate Scaling Test ===")
    
    try:
        wrapper = MultiGPUWrapper(gpu_ids="0,1", distributed=False)
        
        base_lr = 1e-4
        linear_lr = wrapper.scale_learning_rate(base_lr, 'linear')
        sqrt_lr = wrapper.scale_learning_rate(base_lr, 'sqrt')
        none_lr = wrapper.scale_learning_rate(base_lr, 'none')
        
        print(f"✅ Learning rate scaling test:")
        print(f"   Base LR: {base_lr}")
        print(f"   Linear scaling: {linear_lr} (expected: {base_lr * 2})")
        print(f"   Sqrt scaling: {sqrt_lr} (expected: {base_lr * 1.414})")
        print(f"   No scaling: {none_lr} (expected: {base_lr})")
        
        # Verify expected values
        assert abs(linear_lr - base_lr * 2) < 1e-6, "Linear scaling incorrect"
        assert abs(sqrt_lr - base_lr * (2**0.5)) < 1e-6, "Sqrt scaling incorrect"
        assert abs(none_lr - base_lr) < 1e-6, "No scaling incorrect"
        
        return True
    except Exception as e:
        print(f"❌ Learning rate scaling test failed: {e}")
        return False

def test_memory_monitoring():
    """Test GPU memory monitoring"""
    print("\n=== Memory Monitoring Test ===")
    
    try:
        wrapper = MultiGPUWrapper(gpu_ids="0,1", distributed=False)
        
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

def test_training_simulation():
    """Simulate a basic training loop with multi-GPU"""
    print("\n=== Training Simulation Test ===")
    
    try:
        # Create test model and data
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 1)
                
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleModel()
        wrapper = MultiGPUWrapper(gpu_ids="0,1", distributed=False)
        model = wrapper.wrap_model(model)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Simulate training steps
        model.train()
        total_time = 0
        num_steps = 10
        
        print(f"Running {num_steps} training steps...")
        
        for step in range(num_steps):
            start_time = time.time()
            
            # Generate random batch
            batch_size = 8  # Will be split across GPUs
            x = torch.randn(batch_size, 10).to(wrapper.device)
            y = torch.randn(batch_size, 1).to(wrapper.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            step_time = time.time() - start_time
            total_time += step_time
            
            if step % 5 == 0:
                print(f"  Step {step}: loss={loss.item():.4f}, time={step_time:.3f}s")
        
        avg_time = total_time / num_steps
        print(f"✅ Training simulation completed")
        print(f"   Average step time: {avg_time:.3f}s")
        print(f"   Total time: {total_time:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("BEVNeXt-SAM2 Multi-GPU Implementation Test")
    print("=" * 50)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("MultiGPU Wrapper", test_multi_gpu_wrapper),
        ("Model Wrapping", test_model_wrapping),
        ("Learning Rate Scaling", test_learning_rate_scaling),
        ("Memory Monitoring", test_memory_monitoring),
        ("Training Simulation", test_training_simulation),
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
        print("🎉 All tests passed! Multi-GPU implementation is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())