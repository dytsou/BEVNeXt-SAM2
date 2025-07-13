#!/usr/bin/env python3
"""
BEVNeXt-SAM2 Training and Testing Setup Verification Script

This script verifies that the BEVNeXt-SAM2 environment is properly set up
for training and testing, including Docker functionality and dependencies.
"""

import os
import sys
import subprocess
import json
import traceback
from pathlib import Path

def test_imports():
    """Test that key modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will run on CPU")
        
        # Test numpy
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        # Test PIL
        from PIL import Image
        print("✅ PIL imported successfully")
        
        # Test tqdm
        from tqdm import tqdm
        print("✅ tqdm imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_project_structure():
    """Test that essential project files exist"""
    print("\n🔍 Testing project structure...")
    
    essential_files = [
        "tools/train.py",
        "tools/test.py", 
        "training/train_bevnext_sam2.py",
        "integration/bev_sam_fusion.py",
        "integration/sam_enhanced_detector.py",
        "examples/demo_fusion.py",
        "Dockerfile",
        "docker-compose.yml",
        "setup.py",
        "pyproject.toml"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} essential files")
        return False
    else:
        print("\n✅ All essential files present")
        return True

def test_docker_setup():
    """Test Docker configuration"""
    print("\n🔍 Testing Docker setup...")
    
    try:
        # Check if Docker is available
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker available: {result.stdout.strip()}")
        else:
            print("❌ Docker not available")
            return False
            
        # Check Dockerfile
        if os.path.exists('Dockerfile'):
            with open('Dockerfile', 'r') as f:
                dockerfile_content = f.read()
                if 'nvidia/cuda' in dockerfile_content:
                    print("✅ Dockerfile uses NVIDIA CUDA base image")
                else:
                    print("⚠️  Dockerfile doesn't use NVIDIA CUDA base image")
        
        # Check docker-compose
        if os.path.exists('docker-compose.yml'):
            print("✅ docker-compose.yml exists")
        
        return True
        
    except FileNotFoundError:
        print("❌ Docker not found - Docker setup cannot be verified")
        return False
    except Exception as e:
        print(f"❌ Docker setup error: {e}")
        return False

def test_training_config():
    """Test training configuration files"""
    print("\n🔍 Testing training configuration...")
    
    config_files = [
        "training/config_demo.json",
        "training/config_gpu.json", 
        "training/config_cpu_optimized.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    print(f"✅ {config_file} - valid JSON")
                    if 'num_epochs' in config:
                        print(f"   └─ Epochs: {config['num_epochs']}")
                    if 'batch_size' in config:
                        print(f"   └─ Batch size: {config['batch_size']}")
            except Exception as e:
                print(f"❌ {config_file} - invalid JSON: {e}")
        else:
            print(f"❌ {config_file} - missing")
    
    return True

def test_synthetic_data():
    """Test synthetic data generation"""
    print("\n🔍 Testing synthetic data generation...")
    
    try:
        # Add the current directory to Python path
        sys.path.insert(0, '/workspace')
        
        # Import the training module
        from training.train_bevnext_sam2 import SyntheticDataset, get_default_config
        
        # Create a minimal config for testing
        config = get_default_config()
        
        # Create synthetic dataset
        dataset = SyntheticDataset(config, 'train')
        print(f"✅ Synthetic dataset created with {len(dataset)} samples")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"✅ Sample data structure: {list(sample.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data generation error: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🔍 Testing model creation...")
    
    try:
        # Import required modules
        from training.train_bevnext_sam2 import BEVNeXtSAM2Model, get_default_config
        
        # Create config
        config = get_default_config()
        
        # Create model
        model = BEVNeXtSAM2Model(config)
        print("✅ BEVNeXt-SAM2 model created successfully")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        traceback.print_exc()
        return False

def test_integration_modules():
    """Test integration modules"""
    print("\n🔍 Testing integration modules...")
    
    try:
        # Test BEV-SAM fusion import
        from integration.bev_sam_fusion import BEVSAMFusion
        print("✅ BEVSAMFusion imported successfully")
        
        # Test SAM enhanced detector import
        from integration.sam_enhanced_detector import SAMEnhancedBEVDetector
        print("✅ SAMEnhancedBEVDetector imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration modules error: {e}")
        traceback.print_exc()
        return False

def run_mini_training_test():
    """Run a mini training test with 1 epoch"""
    print("\n🔍 Running mini training test...")
    
    try:
        from training.train_bevnext_sam2 import Trainer, get_default_config
        
        # Create minimal config for testing
        config = get_default_config()
        config['num_epochs'] = 1
        config['num_samples'] = {'train': 10, 'val': 5}
        config['batch_size'] = 2
        config['save_interval'] = 1
        
        # Create trainer
        trainer = Trainer(config)
        print("✅ Trainer created successfully")
        
        # Run one training epoch
        print("🏃 Running 1 training epoch...")
        trainer.train()
        print("✅ Mini training test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Mini training test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 BEVNeXt-SAM2 Training and Testing Setup Verification")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Docker Setup", test_docker_setup),
        ("Training Configuration", test_training_config),
        ("Synthetic Data", test_synthetic_data),
        ("Model Creation", test_model_creation),
        ("Integration Modules", test_integration_modules),
        ("Mini Training Test", run_mini_training_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The setup is ready for training and testing.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)