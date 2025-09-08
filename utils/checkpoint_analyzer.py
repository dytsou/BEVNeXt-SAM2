#!/usr/bin/env python3
"""
Enhanced Checkpoint Analysis Utility

This utility provides comprehensive analysis of BEVNeXt-SAM2 checkpoints,
including the new enhanced checkpoint format with resume functionality.

Author: Senior Python Programmer & AI Training Expert
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add training directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "training"))

try:
    from resume_fixes import EnhancedTrainingState, load_checkpoint_with_compatibility
    from checkpoint_manager import CheckpointManager, CheckpointMetadata
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Warning: Enhanced checkpoint features not available")


class CheckpointAnalyzer:
    """Comprehensive checkpoint analyzer for both old and new formats"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_data = None
        self.is_enhanced = False
        self.metadata = None
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint data"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        try:
            self.checkpoint_data = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Check if this is an enhanced checkpoint
            self.is_enhanced = (
                'checkpoint_version' in self.checkpoint_data or
                'training_state' in self.checkpoint_data or
                'global_step' in self.checkpoint_data
            )
            
            print(f"✅ Loaded checkpoint: {self.checkpoint_path}")
            print(f"📊 Enhanced format: {'Yes' if self.is_enhanced else 'No'}")
            
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}")
    
    def analyze_basic_info(self) -> Dict[str, Any]:
        """Analyze basic checkpoint information"""
        info = {}
        
        # File information
        file_stats = self.checkpoint_path.stat()
        info['file_size_mb'] = file_stats.st_size / (1024 * 1024)
        info['file_modified'] = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        
        # Basic checkpoint info
        info['epoch'] = self.checkpoint_data.get('epoch', 'unknown')
        info['timestamp'] = self.checkpoint_data.get('timestamp', 'unknown')
        info['checkpoint_version'] = self.checkpoint_data.get('checkpoint_version', '1.0')
        
        # Step information (handle different naming)
        step_keys = ['global_step', 'step', 'batch_step', 'epoch_step']
        for key in step_keys:
            if key in self.checkpoint_data:
                info[key] = self.checkpoint_data[key]
        
        # Training metrics
        info['best_val_loss'] = self.checkpoint_data.get('best_val_loss', 'unknown')
        
        return info
    
    def analyze_model_info(self) -> Dict[str, Any]:
        """Analyze model-related information"""
        info = {}
        
        if 'model_state_dict' in self.checkpoint_data:
            model_state = self.checkpoint_data['model_state_dict']
            
            # Count parameters
            total_params = 0
            total_size = 0
            layer_info = defaultdict(int)
            
            for name, param in model_state.items():
                if isinstance(param, torch.Tensor):
                    params = param.numel()
                    size = params * param.element_size()
                    
                    total_params += params
                    total_size += size
                    
                    # Group by layer type
                    layer_type = name.split('.')[0] if '.' in name else name
                    layer_info[layer_type] += params
            
            info['total_parameters'] = total_params
            info['total_size_mb'] = total_size / (1024 * 1024)
            info['layer_breakdown'] = dict(layer_info)
            info['num_layers'] = len(model_state)
        
        return info
    
    def analyze_training_state(self) -> Dict[str, Any]:
        """Analyze training state information"""
        info = {}
        
        # Enhanced training state
        if 'training_state' in self.checkpoint_data:
            training_state = self.checkpoint_data['training_state']
            info['enhanced_state'] = training_state
            
            # Calculate training progress
            if 'total_epochs' in training_state and 'epoch' in training_state:
                progress = (training_state['epoch'] + 1) / training_state['total_epochs']
                info['epoch_progress_percent'] = progress * 100
            
            if 'batches_per_epoch' in training_state and 'epoch_step' in training_state:
                epoch_progress = training_state['epoch_step'] / training_state['batches_per_epoch']
                info['current_epoch_progress_percent'] = epoch_progress * 100
        
        # Training statistics
        if 'training_stats' in self.checkpoint_data:
            stats = self.checkpoint_data['training_stats']
            info['training_statistics'] = stats
            
            # Analyze trends if available
            if isinstance(stats, dict):
                for metric, values in stats.items():
                    if isinstance(values, list) and len(values) > 1:
                        info[f'{metric}_trend'] = {
                            'latest': values[-1] if values else None,
                            'best': min(values) if 'loss' in metric.lower() else max(values),
                            'count': len(values)
                        }
        
        # Optimizer state
        if 'optimizer_state_dict' in self.checkpoint_data:
            opt_state = self.checkpoint_data['optimizer_state_dict']
            if 'param_groups' in opt_state:
                info['learning_rate'] = opt_state['param_groups'][0].get('lr', 'unknown')
        
        return info
    
    def analyze_reproducibility(self) -> Dict[str, Any]:
        """Analyze reproducibility information"""
        info = {}
        
        # Random states
        random_states = []
        if 'torch_rng_state' in self.checkpoint_data:
            random_states.append('torch')
        if 'cuda_rng_state' in self.checkpoint_data:
            random_states.append('cuda')
        if 'numpy_rng_state' in self.checkpoint_data:
            random_states.append('numpy')
        if 'python_rng_state' in self.checkpoint_data:
            random_states.append('python')
        
        info['saved_random_states'] = random_states
        info['reproducible'] = len(random_states) >= 2  # At least torch + one other
        
        # Mixed precision scaler
        info['has_scaler_state'] = 'scaler_state_dict' in self.checkpoint_data
        
        return info
    
    def analyze_resume_capability(self) -> Dict[str, Any]:
        """Analyze resume capability"""
        info = {}
        
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        optional_keys = ['scheduler_state_dict', 'training_state', 'global_step']
        
        missing_required = [key for key in required_keys if key not in self.checkpoint_data]
        missing_optional = [key for key in optional_keys if key not in self.checkpoint_data]
        
        info['can_resume'] = len(missing_required) == 0
        info['missing_required'] = missing_required
        info['missing_optional'] = missing_optional
        
        # Resume quality score
        score = 100
        score -= len(missing_required) * 50  # Required keys are critical
        score -= len(missing_optional) * 10   # Optional keys reduce quality
        
        if not self.is_enhanced:
            score -= 20  # Enhanced format provides better resume
        
        info['resume_quality_score'] = max(0, score)
        
        # Estimate data loss on resume
        if 'epoch_step' not in self.checkpoint_data and 'global_step' not in self.checkpoint_data:
            info['estimated_data_loss'] = "Up to 1 epoch (batch position unknown)"
        elif self.is_enhanced:
            info['estimated_data_loss'] = "None (exact position preserved)"
        else:
            info['estimated_data_loss'] = "Minimal (step position available)"
        
        return info
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=" * 80)
        report.append(f"CHECKPOINT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"File: {self.checkpoint_path}")
        report.append(f"Analyzed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic Information
        basic_info = self.analyze_basic_info()
        report.append("📋 BASIC INFORMATION")
        report.append("-" * 40)
        report.append(f"File size: {basic_info['file_size_mb']:.2f} MB")
        report.append(f"Last modified: {basic_info['file_modified']}")
        report.append(f"Checkpoint version: {basic_info['checkpoint_version']}")
        report.append(f"Epoch: {basic_info['epoch']}")
        if 'global_step' in basic_info:
            report.append(f"Global step: {basic_info['global_step']}")
        if 'epoch_step' in basic_info:
            report.append(f"Epoch step: {basic_info['epoch_step']}")
        report.append(f"Best validation loss: {basic_info['best_val_loss']}")
        report.append("")
        
        # Model Information
        model_info = self.analyze_model_info()
        if model_info:
            report.append("🏗️ MODEL INFORMATION")
            report.append("-" * 40)
            report.append(f"Total parameters: {model_info.get('total_parameters', 0):,}")
            report.append(f"Model size: {model_info.get('total_size_mb', 0):.2f} MB")
            report.append(f"Number of layers: {model_info.get('num_layers', 0)}")
            
            # Top layer types by parameter count
            if 'layer_breakdown' in model_info:
                sorted_layers = sorted(model_info['layer_breakdown'].items(), 
                                     key=lambda x: x[1], reverse=True)
                report.append("Top layer types by parameter count:")
                for layer_type, param_count in sorted_layers[:5]:
                    report.append(f"  {layer_type}: {param_count:,} params")
            report.append("")
        
        # Training State
        training_info = self.analyze_training_state()
        if training_info:
            report.append("🎯 TRAINING STATE")
            report.append("-" * 40)
            
            if 'epoch_progress_percent' in training_info:
                report.append(f"Overall progress: {training_info['epoch_progress_percent']:.1f}%")
            
            if 'current_epoch_progress_percent' in training_info:
                report.append(f"Current epoch progress: {training_info['current_epoch_progress_percent']:.1f}%")
            
            if 'learning_rate' in training_info:
                report.append(f"Learning rate: {training_info['learning_rate']}")
            
            # Training trends
            for key, value in training_info.items():
                if key.endswith('_trend') and isinstance(value, dict):
                    metric_name = key.replace('_trend', '')
                    report.append(f"{metric_name}:")
                    report.append(f"  Latest: {value['latest']}")
                    report.append(f"  Best: {value['best']}")
                    report.append(f"  Count: {value['count']}")
            report.append("")
        
        # Reproducibility
        repro_info = self.analyze_reproducibility()
        report.append("🔄 REPRODUCIBILITY")
        report.append("-" * 40)
        report.append(f"Random states saved: {', '.join(repro_info['saved_random_states'])}")
        report.append(f"Fully reproducible: {'Yes' if repro_info['reproducible'] else 'No'}")
        report.append(f"Mixed precision scaler: {'Yes' if repro_info['has_scaler_state'] else 'No'}")
        report.append("")
        
        # Resume Capability
        resume_info = self.analyze_resume_capability()
        report.append("▶️ RESUME CAPABILITY")
        report.append("-" * 40)
        report.append(f"Can resume: {'Yes' if resume_info['can_resume'] else 'No'}")
        report.append(f"Resume quality score: {resume_info['resume_quality_score']}/100")
        report.append(f"Estimated data loss: {resume_info['estimated_data_loss']}")
        
        if resume_info['missing_required']:
            report.append(f"❌ Missing required: {', '.join(resume_info['missing_required'])}")
        
        if resume_info['missing_optional']:
            report.append(f"⚠️  Missing optional: {', '.join(resume_info['missing_optional'])}")
        
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if not self.is_enhanced:
            report.append("• Consider upgrading to enhanced checkpoint format for better resume")
        
        if resume_info['resume_quality_score'] < 80:
            report.append("• Checkpoint quality is suboptimal for reliable resume")
        
        if not repro_info['reproducible']:
            report.append("• Random states incomplete - results may not be reproducible")
        
        if len(repro_info['saved_random_states']) == 0:
            report.append("• No random states saved - training resume will not be deterministic")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_json(self, output_path: str):
        """Export analysis as JSON"""
        analysis = {
            'checkpoint_path': str(self.checkpoint_path),
            'analyzed_at': datetime.now().isoformat(),
            'basic_info': self.analyze_basic_info(),
            'model_info': self.analyze_model_info(),
            'training_state': self.analyze_training_state(),
            'reproducibility': self.analyze_reproducibility(),
            'resume_capability': self.analyze_resume_capability()
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"📄 Analysis exported to: {output_path}")


def analyze_checkpoint_directory(directory: str) -> List[Dict[str, Any]]:
    """Analyze all checkpoints in a directory"""
    directory = Path(directory)
    checkpoint_files = []
    
    # Find checkpoint files
    for pattern in ['*.pth', '*.pt']:
        checkpoint_files.extend(directory.glob(pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {directory}")
        return []
    
    results = []
    
    print(f"📂 Found {len(checkpoint_files)} checkpoint files in {directory}")
    print()
    
    for checkpoint_file in sorted(checkpoint_files):
        try:
            analyzer = CheckpointAnalyzer(str(checkpoint_file))
            basic_info = analyzer.analyze_basic_info()
            resume_info = analyzer.analyze_resume_capability()
            
            result = {
                'file': checkpoint_file.name,
                'size_mb': basic_info['file_size_mb'],
                'epoch': basic_info['epoch'],
                'enhanced': analyzer.is_enhanced,
                'can_resume': resume_info['can_resume'],
                'resume_score': resume_info['resume_quality_score']
            }
            
            # Add step info if available
            for step_key in ['global_step', 'step', 'batch_step']:
                if step_key in basic_info:
                    result[step_key] = basic_info[step_key]
                    break
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed to analyze {checkpoint_file.name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze BEVNeXt-SAM2 checkpoints')
    parser.add_argument('checkpoint', help='Path to checkpoint file or directory')
    parser.add_argument('--output', help='Output file for detailed analysis')
    parser.add_argument('--json', help='Export analysis as JSON')
    parser.add_argument('--directory', action='store_true', 
                       help='Analyze all checkpoints in directory')
    
    args = parser.parse_args()
    
    if args.directory or Path(args.checkpoint).is_dir():
        # Directory analysis
        results = analyze_checkpoint_directory(args.checkpoint)
        
        if results:
            print("\n📊 DIRECTORY SUMMARY")
            print("=" * 60)
            print(f"{'File':<30} {'Size (MB)':<10} {'Epoch':<8} {'Enhanced':<10} {'Resume':<8}")
            print("-" * 60)
            
            for result in results:
                enhanced_str = "Yes" if result['enhanced'] else "No"
                resume_str = f"{result['resume_score']}/100" if result['can_resume'] else "No"
                
                print(f"{result['file']:<30} {result['size_mb']:<10.2f} {result['epoch']:<8} "
                      f"{enhanced_str:<10} {resume_str:<8}")
        
    else:
        # Single checkpoint analysis
        try:
            analyzer = CheckpointAnalyzer(args.checkpoint)
            
            # Generate and display report
            report = analyzer.generate_report()
            print(report)
            
            # Save to output file if specified
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                print(f"\n📄 Report saved to: {args.output}")
            
            # Export as JSON if specified
            if args.json:
                analyzer.export_json(args.json)
                
        except Exception as e:
            print(f"❌ Error analyzing checkpoint: {e}")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
