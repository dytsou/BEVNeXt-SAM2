#!/usr/bin/env python3
"""
Quick Model Validation Script for BEVNeXt-SAM2

This script provides a convenient Python interface for quickly validating 
trained models with common validation tasks and sensible defaults.

Usage:
    python quick_validate.py --checkpoint checkpoints/latest.pth --data-root /path/to/nuscenes
"""

import argparse
import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements() -> bool:
    """Check if required dependencies are available"""
    try:
        import torch
        import numpy as np
        logger.info("✅ Core dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        return False

def validate_inputs(checkpoint_path: str, data_root: str) -> bool:
    """Validate input paths and files"""
    checkpoint_path = Path(checkpoint_path)
    data_root = Path(data_root)
    
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
        return False
    
    if not data_root.exists():
        logger.error(f"❌ Data directory not found: {data_root}")
        return False
    
    # Check for nuScenes structure
    required_dirs = ['samples', 'sweeps', 'maps']
    missing_dirs = []
    for dir_name in required_dirs:
        if not (data_root / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        logger.warning(f"⚠️  Missing nuScenes directories: {missing_dirs}")
        logger.warning("This may indicate an incomplete dataset setup")
    
    logger.info("✅ Input validation passed")
    return True

def run_quick_validation(
    checkpoint_path: str,
    data_root: str,
    output_dir: str = "outputs/quick_validation",
    max_samples: int = 100,
    device: str = "auto",
    generate_viz: bool = False,
    run_nuscenes_eval: bool = False,
    verbose: bool = False
) -> Dict:
    """Run quick model validation"""
    
    logger.info("🚀 Starting BEVNeXt-SAM2 quick validation...")
    logger.info(f"Configuration:")
    logger.info(f"  └─ Checkpoint: {checkpoint_path}")
    logger.info(f"  └─ Data root: {data_root}")
    logger.info(f"  └─ Output dir: {output_dir}")
    logger.info(f"  └─ Max samples: {max_samples}")
    logger.info(f"  └─ Device: {device}")
    logger.info(f"  └─ Generate viz: {generate_viz}")
    logger.info(f"  └─ nuScenes eval: {run_nuscenes_eval}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build validation command
    cmd = [
        sys.executable, "validation/validate_model.py",
        "--checkpoint", checkpoint_path,
        "--data-root", data_root,
        "--output-dir", output_dir,
        "--max-samples", str(max_samples)
    ]
    
    if run_nuscenes_eval:
        cmd.append("--run-nuscenes-eval")
    
    if generate_viz:
        cmd.append("--generate-viz")
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    # Run validation
    logger.info("🔄 Running validation...")
    if verbose:
        logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=not verbose,
            text=True,
            check=True
        )
        
        logger.info("✅ Validation completed successfully!")
        
        # Try to load and display results
        metrics_file = Path(output_dir) / "validation_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                logger.info("📊 Key validation metrics:")
                key_metrics = ['mean_iou', 'classification_accuracy', 'max_iou']
                for metric in key_metrics:
                    if metric in metrics:
                        logger.info(f"  └─ {metric}: {metrics[metric]:.4f}")
                
                return metrics
                
            except Exception as e:
                logger.warning(f"Could not parse metrics file: {e}")
        
        return {"status": "completed"}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Validation failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during validation: {e}")
        raise

def run_dataset_validation(
    data_root: str,
    version: str = "v1.0-trainval",
    output_dir: str = "outputs/dataset_validation",
    verbose: bool = False
) -> bool:
    """Run dataset integrity validation"""
    
    logger.info("🔍 Running dataset validation...")
    
    cmd = [
        sys.executable, "validation/nuscenes_validator.py",
        "--data-root", data_root,
        "--version", version,
        "--output-dir", output_dir
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True
        )
        
        logger.info("✅ Dataset validation completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Dataset validation failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Quick BEVNeXt-SAM2 Model Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_validate.py --checkpoint checkpoints/latest.pth --data-root /data/nuscenes
  python quick_validate.py --checkpoint models/best.pth --full --verbose
  python quick_validate.py --dataset-only --data-root /data/nuscenes
        """
    )
    
    # Validation options
    parser.add_argument('--checkpoint', '-c', 
                       default='checkpoints/latest.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data-root', '-d',
                       default='data/nuscenes', 
                       help='Path to nuScenes dataset')
    parser.add_argument('--output-dir', '-o',
                       default='outputs/quick_validation',
                       help='Output directory for results')
    parser.add_argument('--max-samples', '-s',
                       type=int, default=100,
                       help='Maximum samples to validate')
    parser.add_argument('--device',
                       default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for validation')
    
    # Validation modes
    parser.add_argument('--full',
                       action='store_true',
                       help='Run full validation (nuScenes eval + visualizations)')
    parser.add_argument('--viz',
                       action='store_true', 
                       help='Generate visualizations')
    parser.add_argument('--nuscenes-eval',
                       action='store_true',
                       help='Run official nuScenes evaluation')
    parser.add_argument('--dataset-only',
                       action='store_true',
                       help='Only validate dataset integrity')
    
    # General options
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Adjust settings for full validation
    if args.full:
        args.nuscenes_eval = True
        args.viz = True
        args.max_samples = 500
    
    try:
        # Check requirements
        if not check_requirements():
            return 1
        
        # Validate inputs (skip checkpoint check for dataset-only mode)
        if not args.dataset_only:
            if not validate_inputs(args.checkpoint, args.data_root):
                return 1
        else:
            if not Path(args.data_root).exists():
                logger.error(f"❌ Data directory not found: {args.data_root}")
                return 1
        
        if args.dataset_only:
            # Run only dataset validation
            success = run_dataset_validation(
                data_root=args.data_root,
                output_dir=args.output_dir,
                verbose=args.verbose
            )
            return 0 if success else 1
        
        else:
            # Run model validation
            metrics = run_quick_validation(
                checkpoint_path=args.checkpoint,
                data_root=args.data_root,
                output_dir=args.output_dir,
                max_samples=args.max_samples,
                device=args.device,
                generate_viz=args.viz,
                run_nuscenes_eval=args.nuscenes_eval,
                verbose=args.verbose
            )
            
            logger.info(f"📁 Results saved to: {args.output_dir}")
            logger.info("🎉 Validation completed successfully!")
            return 0
    
    except KeyboardInterrupt:
        logger.warning("❌ Validation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
