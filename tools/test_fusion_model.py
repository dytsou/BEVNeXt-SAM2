#!/usr/bin/env python3
"""
BEVNeXt-SAM2 Fusion Model Testing Script

Comprehensive testing and evaluation pipeline for the fusion model.
Includes performance benchmarking, visualization, and result analysis.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.bevnext_sam2_fusion_model import BEVNeXtSAM2FusionModel
from tools.fusion_dataset import FusionDataset, SyntheticFusionDataset, create_fusion_dataloader
from tools.fusion_evaluator import FusionEvaluator
from tools.fusion_visualizer import FusionVisualizer


class FusionModelTester:
    """
    Comprehensive tester for BEVNeXt-SAM2 fusion model.
    
    Features:
    - Model loading and validation
    - Performance benchmarking
    - Comprehensive evaluation
    - Visualization generation
    - Result analysis and reporting
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.test_loader = None
        self.evaluator = FusionEvaluator(args.eval_config)
        self.visualizer = FusionVisualizer(self.output_dir / 'visualizations')
        
        print(f"🧪 Fusion Model Tester initialized")
        print(f"   Device: {self.device}")
        print(f"   Output directory: {self.output_dir}")
        
    def load_model(self):
        """Load the fusion model from checkpoint."""
        print("🔄 Loading fusion model...")
        
        # Build model
        self.model = BEVNeXtSAM2FusionModel(
            bev_config=self.args.bev_config,
            sam2_config=self.args.sam2_config,
            sam2_checkpoint=self.args.sam2_checkpoint,
            fusion_mode=self.args.fusion_mode,
            device=self.device
        ).to(self.device)
        
        # Load checkpoint
        if self.args.checkpoint:
            print(f"Loading checkpoint: {self.args.checkpoint}")
            checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded successfully!")
        print(f"   Total parameters: {total_params:,}")
        
    def setup_data(self):
        """Setup test dataset and data loader."""
        print("📊 Setting up test dataset...")
        
        if self.args.use_synthetic:
            # Use synthetic data for testing
            self.test_dataset = SyntheticFusionDataset(
                num_samples=self.args.num_samples,
                split='test'
            )
        else:
            # Use real dataset
            self.test_dataset = FusionDataset(
                data_config=self.args.data_config,
                split='test',
                max_samples=self.args.num_samples
            )
        
        self.test_loader = create_fusion_dataloader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"   Test samples: {len(self.test_dataset)}")
        print(f"   Batch size: {self.args.batch_size}")
        
    def run_inference(self) -> Dict:
        """Run inference on test dataset and collect results."""
        print("\n🚀 Running inference...")
        
        all_predictions = []
        all_targets = []
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.test_loader,
                desc="Testing",
                total=len(self.test_loader)
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                images = batch['images'].to(self.device)
                img_metas = batch['img_metas']
                
                # Measure inference time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                # Forward pass
                predictions = self.model.forward_test(images, img_metas)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Record metrics
                batch_time = (end_time - start_time) * 1000  # Convert to ms
                inference_times.append(batch_time)
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                    memory_usage.append(memory_mb)
                
                # Store results
                all_predictions.extend(predictions)
                
                # Create targets from batch
                for i in range(len(img_metas)):
                    target = {
                        'gt_bboxes_3d': batch['gt_bboxes_3d'][i],
                        'gt_labels_3d': batch['gt_labels_3d'][i],
                        'img_metas': img_metas[i]
                    }
                    if batch.get('gt_masks') is not None:
                        target['gt_masks'] = batch['gt_masks'][i]
                    all_targets.append(target)
                
                # Update progress
                progress_bar.set_postfix({
                    'avg_time': f'{np.mean(inference_times[-10:]):.1f}ms',
                    'memory': f'{memory_usage[-1]:.0f}MB' if memory_usage else 'N/A'
                })
                
                # Save sample visualizations
                if batch_idx % self.args.vis_interval == 0:
                    self.save_sample_visualization(
                        batch, predictions[:len(img_metas)], batch_idx
                    )
        
        # Compile results
        results = {
            'predictions': all_predictions,
            'targets': all_targets,
            'inference_times': inference_times,
            'memory_usage': memory_usage,
            'total_samples': len(all_predictions)
        }
        
        return results
        
    def evaluate_results(self, results: Dict) -> Dict:
        """Evaluate model performance on test results."""
        print("\n📈 Evaluating results...")
        
        # Comprehensive evaluation
        eval_metrics = self.evaluator.evaluate(
            results['predictions'],
            results['targets']
        )
        
        # Performance metrics
        inference_times = results['inference_times']
        memory_usage = results['memory_usage']
        
        performance_metrics = {
            'avg_inference_time_ms': np.mean(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'fps': 1000 / np.mean(inference_times),  # Frames per second
            'total_samples': results['total_samples']
        }
        
        if memory_usage:
            performance_metrics.update({
                'avg_memory_usage_mb': np.mean(memory_usage),
                'max_memory_usage_mb': np.max(memory_usage)
            })
        
        # Combine all metrics
        final_metrics = {
            **eval_metrics,
            **performance_metrics
        }
        
        return final_metrics
        
    def save_sample_visualization(
        self,
        batch: Dict,
        predictions: List[Dict],
        batch_idx: int
    ):
        """Save visualization for a sample batch."""
        
        # Visualize first sample in batch
        sample_idx = 0
        if sample_idx < len(predictions):
            save_path = self.output_dir / 'visualizations' / f'test_batch_{batch_idx}_sample_{sample_idx}.png'
            
            self.visualizer.visualize_sample(
                images=batch['images'][sample_idx],
                predictions=predictions[sample_idx],
                targets={
                    'gt_bboxes_3d': batch['gt_bboxes_3d'][sample_idx],
                    'gt_labels_3d': batch['gt_labels_3d'][sample_idx],
                    'gt_masks': batch.get('gt_masks', [None])[sample_idx]
                },
                img_meta=batch['img_metas'][sample_idx],
                save_path=save_path
            )
    
    def generate_comprehensive_report(self, metrics: Dict):
        """Generate comprehensive evaluation report."""
        print("\n📄 Generating comprehensive report...")
        
        # Save metrics to JSON
        metrics_path = self.output_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = value.item()
                else:
                    serializable_metrics[key] = value
            json.dump(serializable_metrics, f, indent=2)
        
        # Generate evaluation plots
        self.evaluator.save_evaluation_report(
            metrics,
            str(self.output_dir / 'evaluation_report')
        )
        
        # Generate text report
        self.generate_text_report(metrics)
        
        # Export to HTML
        self.visualizer.export_to_html(
            metrics,
            self.output_dir / 'results.html'
        )
        
        print(f"✅ Comprehensive report saved to {self.output_dir}")
        
    def generate_text_report(self, metrics: Dict):
        """Generate human-readable text report."""
        
        report_lines = [
            "BEVNeXt-SAM2 Fusion Model Test Report",
            "=" * 50,
            "",
            f"Test Configuration:",
            f"  - Model: {self.args.fusion_mode} fusion",
            f"  - Test samples: {metrics.get('total_samples', 'N/A')}",
            f"  - Batch size: {self.args.batch_size}",
            f"  - Device: {self.device}",
            "",
            "Performance Metrics:",
            f"  - Average inference time: {metrics.get('avg_inference_time_ms', 0):.2f} ms",
            f"  - FPS: {metrics.get('fps', 0):.2f}",
            f"  - Memory usage: {metrics.get('avg_memory_usage_mb', 0):.1f} MB",
            "",
            "3D Detection Results:",
            f"  - mAP: {metrics.get('3d_mAP_3d', 0):.4f}",
            f"  - NDS: {metrics.get('3d_NDS', 0):.4f}",
        ]
        
        # Add class-specific AP if available
        for class_name in ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']:
            ap_key = f'3d_AP_{class_name}@0.5'
            if ap_key in metrics:
                report_lines.append(f"  - AP {class_name}: {metrics[ap_key]:.4f}")
        
        report_lines.extend([
            "",
            "2D Segmentation Results:",
            f"  - IoU: {metrics.get('2d_IoU', 0):.4f}",
            f"  - F1 Score: {metrics.get('2d_F1', 0):.4f}",
            f"  - Precision: {metrics.get('2d_Precision', 0):.4f}",
            f"  - Recall: {metrics.get('2d_Recall', 0):.4f}",
            "",
            "Cross-Modal Consistency:",
            f"  - Spatial alignment: {metrics.get('consistency_spatial_alignment', 0):.4f}",
            f"  - Detection-segmentation: {metrics.get('consistency_det_seg_consistency', 0):.4f}",
            "",
            f"Overall Score: {metrics.get('overall_score', 0):.4f}",
            ""
        ])
        
        # Save text report
        report_path = self.output_dir / 'test_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
    def run_benchmark(self):
        """Run performance benchmark."""
        print("\n⚡ Running performance benchmark...")
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(5):
                dummy_input = torch.randn(1, 6, 3, 448, 800).to(self.device)
                dummy_metas = [{'cam_name': f'cam_{i}', 'img_shape': (448, 800, 3)} for i in range(6)]
                _ = self.model.forward_test(dummy_input, dummy_metas)
        
        # Benchmark different batch sizes
        batch_sizes = [1, 2, 4] if torch.cuda.is_available() else [1]
        benchmark_results = {}
        
        for batch_size in batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            times = []
            memory_usage = []
            
            print(f"Benchmarking batch size {batch_size}...")
            
            with torch.no_grad():
                for _ in range(20):  # 20 iterations for stable measurement
                    # Create dummy batch
                    dummy_input = torch.randn(batch_size, 6, 3, 448, 800).to(self.device)
                    dummy_metas = [
                        [{'cam_name': f'cam_{i}', 'img_shape': (448, 800, 3)} for i in range(6)]
                        for _ in range(batch_size)
                    ]
                    
                    # Measure time
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.time()
                    
                    _ = self.model.forward_test(dummy_input, dummy_metas)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    
                    batch_time = (end_time - start_time) * 1000  # ms
                    times.append(batch_time)
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                        memory_usage.append(memory_mb)
            
            benchmark_results[batch_size] = {
                'avg_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'avg_time_per_sample_ms': np.mean(times) / batch_size,
                'fps': 1000 * batch_size / np.mean(times),
                'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0
            }
        
        # Save benchmark results
        benchmark_path = self.output_dir / 'benchmark_results.json'
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Print benchmark summary
        print("\nBenchmark Results:")
        for batch_size, results in benchmark_results.items():
            print(f"  Batch size {batch_size}:")
            print(f"    - Avg time: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
            print(f"    - Per sample: {results['avg_time_per_sample_ms']:.2f} ms")
            print(f"    - FPS: {results['fps']:.2f}")
            print(f"    - Memory: {results['avg_memory_mb']:.1f} MB")
    
    def run_tests(self):
        """Run comprehensive testing pipeline."""
        try:
            # Load model
            self.load_model()
            
            # Setup data
            self.setup_data()
            
            # Run inference
            results = self.run_inference()
            
            # Evaluate results
            metrics = self.evaluate_results(results)
            
            # Generate report
            self.generate_comprehensive_report(metrics)
            
            # Run benchmark if requested
            if self.args.benchmark:
                self.run_benchmark()
            
            print(f"\n🎉 Testing completed successfully!")
            print(f"Results saved to: {self.output_dir}")
            
            # Print key metrics
            print(f"\nKey Results:")
            print(f"  - Overall Score: {metrics.get('overall_score', 0):.4f}")
            print(f"  - 3D mAP: {metrics.get('3d_mAP_3d', 0):.4f}")
            print(f"  - 2D IoU: {metrics.get('2d_IoU', 0):.4f}")
            print(f"  - Average FPS: {metrics.get('fps', 0):.2f}")
            
        except Exception as e:
            print(f"❌ Testing failed: {str(e)}")
            raise


def parse_args():
    parser = argparse.ArgumentParser(description='Test BEVNeXt-SAM2 Fusion Model')
    
    # Model configuration
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--bev-config', type=str, required=True,
                       help='BEVNeXt configuration file')
    parser.add_argument('--sam2-config', type=str, required=True,
                       help='SAM2 configuration file')
    parser.add_argument('--sam2-checkpoint', type=str,
                       help='SAM2 checkpoint file')
    parser.add_argument('--fusion-mode', type=str, default='feature_fusion',
                       choices=['feature_fusion', 'late_fusion', 'multi_scale'],
                       help='Fusion mode')
    
    # Data configuration  
    parser.add_argument('--data-config', type=str,
                       help='Test dataset configuration file')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Test batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Evaluation configuration
    parser.add_argument('--eval-config', type=str,
                       help='Evaluation configuration file')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory for test results')
    
    # Testing options
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--vis-interval', type=int, default=10,
                       help='Visualization interval')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create tester and run tests
    tester = FusionModelTester(args)
    tester.run_tests()


if __name__ == '__main__':
    main()