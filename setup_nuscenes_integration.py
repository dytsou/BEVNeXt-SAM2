#!/usr/bin/env python3
"""
Complete nuScenes v1.0 Integration Setup Script

This script sets up the complete nuScenes v1.0 integration for BEVNeXt-SAM2 training,
including dataset validation, training configuration, and analysis tools.

Usage:
    python setup_nuscenes_integration.py --data-root /path/to/nuscenes --action [setup|validate|train|analyze]
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NuScenesIntegrationManager:
    """Manages complete nuScenes integration setup and workflows"""
    
    def __init__(self, data_root: str, version: str = "v1.0-trainval"):
        self.data_root = Path(data_root)
        self.version = version
        self.project_root = Path(__file__).parent
        
    def setup_environment(self) -> bool:
        """Setup environment for nuScenes integration"""
        logger.info("Setting up nuScenes integration environment...")
        
        # Check if nuScenes data directory exists
        if not self.data_root.exists():
            logger.error(f"nuScenes data directory not found: {self.data_root}")
            logger.info("Please ensure nuScenes dataset is extracted to the correct location.")
            return False
        
        # Check for version directory
        version_dir = self.data_root / self.version
        if not version_dir.exists():
            # Check for alternative versions
            available_versions = [d.name for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith('v1.0')]
            if available_versions:
                self.version = available_versions[0]
                logger.info(f"Using available version: {self.version}")
            else:
                logger.error("No nuScenes version directories found")
                return False
        
        # Install dependencies
        if not self._install_dependencies():
            return False
        
        # Verify installation
        if not self._verify_installation():
            return False
        
        logger.info("Environment setup complete!")
        return True
    
    def _install_dependencies(self) -> bool:
        """Install required dependencies"""
        logger.info("Installing nuScenes dependencies...")
        
        dependencies = [
            "nuscenes-devkit",
            "pyquaternion",
            "shapely",
            "matplotlib",
            "seaborn",
            "pandas",
            "tqdm"
        ]
        
        for dep in dependencies:
            try:
                logger.info(f"   Installing {dep}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to install {dep}, continuing...")
        
        return True
    
    def _verify_installation(self) -> bool:
        """Verify nuScenes installation"""
        try:
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version=self.version, dataroot=str(self.data_root), verbose=False)
            logger.info(f"nuScenes {self.version} verified ({len(nusc.sample)} samples)")
            return True
        except Exception as e:
            logger.error(f"nuScenes verification failed: {e}")
            return False
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Run comprehensive dataset validation"""
        logger.info("Running dataset validation...")
        
        try:
            # Import validation module
            sys.path.insert(0, str(self.project_root / "validation"))
            from nuscenes_validator import run_complete_validation
            
            # Run validation
            report = run_complete_validation(
                data_root=str(self.data_root),
                version=self.version,
                output_dir="validation_reports",
                verbose=False
            )
            
            overall_score = report.quality_scores['overall']
            if overall_score >= 90:
                logger.info("Dataset validation PASSED - Excellent quality")
            elif overall_score >= 70:
                logger.info("Dataset validation PASSED - Good quality")
            else:
                logger.warning("Dataset validation passed with issues")
            
            return {
                'status': 'success',
                'quality_scores': report.quality_scores,
                'recommendations': report._generate_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Run comprehensive dataset analysis"""
        logger.info("Running dataset analysis...")
        
        try:
            # Import analysis module
            sys.path.insert(0, str(self.project_root / "utils"))
            from nuscenes_data_analysis import NuScenesDataAnalyzer
            
            # Create analyzer and run analysis
            analyzer = NuScenesDataAnalyzer(str(self.data_root), self.version, verbose=False)
            results = analyzer.run_complete_analysis()
            
            # Print summary
            analyzer.print_summary()
            
            # Generate visualizations
            analyzer.generate_visualizations("analysis_output")
            
            # Save results
            analyzer.save_results(f"analysis_output/nuscenes_analysis_{self.version}.json")
            
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def setup_training(self, config_type: str = "auto") -> Dict[str, Any]:
        """Setup training configuration"""
        logger.info("Setting up training configuration...")
        
        # Detect GPU capabilities
        gpu_memory = self._detect_gpu_memory()
        
        # Select appropriate configuration
        if config_type == "auto":
            if gpu_memory >= 20:
                config_name = "high_end"
            elif gpu_memory >= 12:
                config_name = "mid_range"
            elif gpu_memory >= 10:
                config_name = "rtx_2080ti"
            elif gpu_memory >= 6:
                config_name = "low_end"
            else:
                config_name = "ultra_low"
        else:
            config_name = config_type
        
        # Create training configuration
        config = self._create_training_config(config_name, gpu_memory)
        
        # Save configuration
        config_dir = self.project_root / "training" / "configs"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / f"nuscenes_{config_name}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training configuration saved: {config_file}")
        logger.info(f"   GPU Memory: {gpu_memory:.1f}GB")
        logger.info(f"   Configuration: {config_name}")
        logger.info(f"   Batch Size: {config['batch_size']}")
        logger.info(f"   Model Size: {config['d_model']}d, {config['num_transformer_layers']} layers")
        
        return {
            'status': 'success',
            'config_file': str(config_file),
            'config': config,
            'gpu_memory': gpu_memory,
            'config_type': config_name
        }
    
    def _detect_gpu_memory(self) -> float:
        """Detect GPU memory capacity"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return gpu_memory
        except:
            pass
        return 0.0
    
    def _create_training_config(self, config_type: str, gpu_memory: float) -> Dict[str, Any]:
        """Create training configuration based on GPU capabilities"""
        
        base_config = {
            'data_root': str(self.data_root),
            'nuscenes_version': self.version,
            'num_classes': 23,  # nuScenes categories
            'camera_names': ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'],
            'use_camera': True,
            'use_lidar': False,  # Disabled for memory efficiency
            'use_radar': False,  # Disabled for memory efficiency
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 50,
            'gradient_checkpointing': True,
            'use_augmentation': True,
            'loss_weights': {'cls': 1.0, 'reg': 5.0, 'conf': 1.0}
        }
        
        # Configuration variants
        configs = {
            'ultra_low': {
                'd_model': 64, 'nhead': 4, 'num_transformer_layers': 2,
                'num_queries': 50, 'bev_size': [32, 32], 'image_size': [128, 128],
                'batch_size': 1, 'num_workers': 1, 'memory_fraction': 0.5
            },
            'low_end': {
                'd_model': 96, 'nhead': 6, 'num_transformer_layers': 3,
                'num_queries': 100, 'bev_size': [32, 32], 'image_size': [160, 160],
                'batch_size': 1, 'num_workers': 2, 'memory_fraction': 0.6
            },
            'rtx_2080ti': {
                'd_model': 128, 'nhead': 8, 'num_transformer_layers': 4,
                'num_queries': 150, 'bev_size': [48, 48], 'image_size': [224, 224],
                'batch_size': 1, 'num_workers': 2, 'memory_fraction': 0.7
            },
            'mid_range': {
                'd_model': 256, 'nhead': 8, 'num_transformer_layers': 6,
                'num_queries': 300, 'bev_size': [64, 64], 'image_size': [224, 224],
                'batch_size': 2, 'num_workers': 4, 'memory_fraction': 0.8
            },
            'high_end': {
                'd_model': 512, 'nhead': 16, 'num_transformer_layers': 8,
                'num_queries': 500, 'bev_size': [128, 128], 'image_size': [224, 224],
                'batch_size': 4, 'num_workers': 8, 'memory_fraction': 0.9
            }
        }
        
        # Merge base config with variant
        config = {**base_config, **configs.get(config_type, configs['rtx_2080ti'])}
        config['output_dir'] = f'/workspace/outputs/training_nuscenes_{config_type}'
        
        return config
    
    def start_training(self, config_file: str = None) -> bool:
        """Start training with nuScenes dataset"""
        logger.info("Starting nuScenes training...")
        
        try:
            # Import training module
            sys.path.insert(0, str(self.project_root / "training"))
            from train_bevnext_sam2_nuscenes import main as train_main
            
            # Setup arguments
            sys.argv = ['train_bevnext_sam2_nuscenes.py']
            if config_file:
                sys.argv.extend(['--config', config_file])
            sys.argv.extend(['--data-root', str(self.data_root)])
            sys.argv.extend(['--mixed-precision'])
            
            # Start training
            train_main()
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def print_status(self):
        """Print current integration status"""
        print("\n" + "="*80)
        print("NUSCENES INTEGRATION STATUS")
        print("="*80)
        
        # Check data directory
        if self.data_root.exists():
            print(f"Data directory found: {self.data_root}")
        else:
            print(f"Data directory not found: {self.data_root}")
        
        # Check version directory
        version_dir = self.data_root / self.version
        if version_dir.exists():
            print(f"Version directory found: {self.version}")
        else:
            available_versions = [d.name for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith('v1.0')]
            if available_versions:
                print(f"Version {self.version} not found, available: {available_versions}")
            else:
                print(f"No version directories found")
        
        # Check dependencies
        try:
            from nuscenes.nuscenes import NuScenes
            print("nuScenes devkit available")
        except ImportError:
            print("nuScenes devkit not available")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU available: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
            else:
                print("No GPU available")
        except:
            print("PyTorch not available")
        
        print("="*80)
    
    def run_complete_workflow(self) -> bool:
        """Run complete integration workflow"""
        logger.info("Running complete nuScenes integration workflow...")
        
        # Step 1: Setup environment
        if not self.setup_environment():
            return False
        
        # Step 2: Validate dataset
        validation_result = self.validate_dataset()
        if validation_result['status'] != 'success':
            logger.warning("Dataset validation had issues, but continuing...")
        
        # Step 3: Analyze dataset
        analysis_result = self.analyze_dataset()
        if analysis_result['status'] != 'success':
            logger.warning("Dataset analysis failed, but continuing...")
        
        # Step 4: Setup training
        training_setup = self.setup_training("auto")
        if training_setup['status'] != 'success':
            return False
        
        logger.info("Complete integration workflow finished!")
        logger.info("\nNext steps:")
        logger.info(f"1. Review training config: {training_setup['config_file']}")
        logger.info("2. Start training with: python setup_nuscenes_integration.py --action train")
        logger.info("3. Monitor training progress in outputs directory")
        
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='nuScenes Integration Setup')
    parser.add_argument('--data-root', default='data/nuscenes', help='Path to nuScenes dataset')
    parser.add_argument('--version', default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--action', choices=['setup', 'validate', 'analyze', 'train', 'status', 'complete'], 
                       default='complete', help='Action to perform')
    parser.add_argument('--config', help='Training config file')
    parser.add_argument('--config-type', choices=['ultra_low', 'low_end', 'rtx_2080ti', 'mid_range', 'high_end', 'auto'],
                       default='auto', help='Training configuration type')
    
    args = parser.parse_args()
    
    # Create integration manager
    manager = NuScenesIntegrationManager(args.data_root, args.version)
    
    # Execute requested action
    if args.action == 'status':
        manager.print_status()
    
    elif args.action == 'setup':
        success = manager.setup_environment()
        sys.exit(0 if success else 1)
    
    elif args.action == 'validate':
        result = manager.validate_dataset()
        sys.exit(0 if result['status'] == 'success' else 1)
    
    elif args.action == 'analyze':
        result = manager.analyze_dataset()
        sys.exit(0 if result['status'] == 'success' else 1)
    
    elif args.action == 'train':
        config_file = args.config
        if not config_file:
            # Setup training first
            setup_result = manager.setup_training(args.config_type)
            if setup_result['status'] == 'success':
                config_file = setup_result['config_file']
            else:
                logger.error("Failed to setup training configuration")
                sys.exit(1)
        
        success = manager.start_training(config_file)
        sys.exit(0 if success else 1)
    
    elif args.action == 'complete':
        success = manager.run_complete_workflow()
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 