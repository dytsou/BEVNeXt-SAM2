#!/usr/bin/env python3
"""
BEVNeXt-SAM2 Fusion Model - Complete Demo

This script demonstrates the complete pipeline for training, validating, and testing
the BEVNeXt-SAM2 fusion model. It includes:

1. Model setup and configuration
2. Data preparation
3. Training with validation
4. Testing and evaluation  
5. Comprehensive visualization
6. Result analysis

Usage:
    python examples/demo_fusion_complete.py --mode train
    python examples/demo_fusion_complete.py --mode test --checkpoint work_dirs/fusion_demo/checkpoints/best.pth
    python examples/demo_fusion_complete.py --mode demo --use-synthetic
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.bevnext_sam2_fusion_model import BEVNeXtSAM2FusionModel
from tools.fusion_dataset import SyntheticFusionDataset, create_fusion_dataloader
from tools.fusion_evaluator import FusionEvaluator
from tools.fusion_visualizer import FusionVisualizer
from tools.train_fusion_model import FusionTrainer
from tools.test_fusion_model import FusionModelTester


class FusionDemo:
    """
    Complete demonstration of BEVNeXt-SAM2 fusion model capabilities.
    
    This class provides a unified interface for demonstrating all aspects
    of the fusion model including training, testing, and visualization.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup demo directory
        self.demo_dir = Path(args.demo_dir)
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.evaluator = FusionEvaluator()
        self.visualizer = FusionVisualizer(self.demo_dir / 'visualizations')
        
        print(f"🎬 BEVNeXt-SAM2 Fusion Demo")
        print(f"   Mode: {args.mode}")
        print(f"   Device: {self.device}")
        print(f"   Demo directory: {self.demo_dir}")
        
    def create_demo_configs(self):
        """Create demonstration configuration files."""
        print("📝 Creating demo configurations...")
        
        # BEV configuration (simplified)
        bev_config = {
            'model': {
                'type': 'BEVNext',
                'backbone': {
                    'type': 'ResNet',
                    'depth': 50
                },
                'neck': {
                    'type': 'FPN',
                    'in_channels': [256, 512, 1024, 2048],
                    'out_channels': 256
                },
                'pts_bbox_head': {
                    'type': 'Anchor3DHead',
                    'num_classes': 7,
                    'in_channels': 256
                }
            },
            'train_cfg': {},
            'test_cfg': {}
        }
        
        # SAM2 configuration
        sam2_config = "sam2_module/configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # Data configuration for synthetic data
        data_config = {
            'type': 'synthetic',
            'num_samples': self.args.num_samples,
            'img_size': (448, 800),
            'camera_names': ['front', 'front_left', 'front_right', 'back', 'back_left', 'back_right'],
            'class_names': ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']
        }
        
        # Save configurations
        config_dir = self.demo_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        
        with open(config_dir / 'bev_demo.json', 'w') as f:
            json.dump(bev_config, f, indent=2)
            
        with open(config_dir / 'data_demo.json', 'w') as f:
            json.dump(data_config, f, indent=2)
        
        return {
            'bev_config': config_dir / 'bev_demo.json',
            'sam2_config': sam2_config,
            'data_config': config_dir / 'data_demo.json'
        }
    
    def setup_demo_model(self, configs: Dict[str, Path]):
        """Setup fusion model for demonstration."""
        print("🔧 Setting up fusion model...")
        
        # Create model
        self.model = BEVNeXtSAM2FusionModel(
            bev_config=configs['bev_config'],
            sam2_config=configs['sam2_config'],
            sam2_checkpoint=self.args.sam2_checkpoint,
            fusion_mode=self.args.fusion_mode,
            device=self.device
        ).to(self.device)
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Model created successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def setup_demo_data(self, data_config: Path):
        """Setup demonstration dataset."""
        print("📊 Setting up demo dataset...")
        
        # Create synthetic datasets
        train_dataset = SyntheticFusionDataset(
            num_samples=max(self.args.num_samples // 2, 10),
            split='train',
            load_masks=True
        )
        
        val_dataset = SyntheticFusionDataset(
            num_samples=max(self.args.num_samples // 4, 5),
            split='val',
            load_masks=True
        )
        
        test_dataset = SyntheticFusionDataset(
            num_samples=max(self.args.num_samples // 4, 5),
            split='test',
            load_masks=True
        )
        
        # Create data loaders
        train_loader = create_fusion_dataloader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0
        )
        
        val_loader = create_fusion_dataloader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        test_loader = create_fusion_dataloader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }
    
    def run_demo_training(self, model, data_loaders):
        """Run demonstration training."""
        print("\n🎯 Running demo training...")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Training configuration
        num_epochs = self.args.epochs
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        
        # Training history for visualization
        train_losses = []
        val_losses = []
        val_metrics = []
        
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 5:  # Limit batches for demo
                    break
                    
                # Move data to device
                images = batch['images'].to(self.device)
                img_metas = batch['img_metas']
                gt_bboxes_3d = [bbox.to(self.device) for bbox in batch['gt_bboxes_3d']]
                gt_labels_3d = [labels.to(self.device) for labels in batch['gt_labels_3d']]
                gt_masks = batch.get('gt_masks')
                if gt_masks is not None:
                    gt_masks = [mask.to(self.device) for mask in gt_masks]
                
                # Forward pass
                optimizer.zero_grad()
                losses = model.forward_train(
                    images=images,
                    img_metas=img_metas,
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    gt_masks=gt_masks
                )
                
                # Backward pass
                loss = losses['total']
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_train_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            val_predictions = []
            val_targets = []
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 3:  # Limit batches for demo
                        break
                        
                    # Move data to device
                    images = batch['images'].to(self.device)
                    img_metas = batch['img_metas']
                    gt_bboxes_3d = [bbox.to(self.device) for bbox in batch['gt_bboxes_3d']]
                    gt_labels_3d = [labels.to(self.device) for labels in batch['gt_labels_3d']]
                    gt_masks = batch.get('gt_masks')
                    if gt_masks is not None:
                        gt_masks = [mask.to(self.device) for mask in gt_masks]
                    
                    # Forward pass
                    losses = model.forward_train(
                        images=images,
                        img_metas=img_metas,
                        gt_bboxes_3d=gt_bboxes_3d,
                        gt_labels_3d=gt_labels_3d,
                        gt_masks=gt_masks
                    )
                    
                    predictions = model.forward_test(images, img_metas)
                    
                    epoch_val_loss += losses['total'].item()
                    num_val_batches += 1
                    
                    # Store for evaluation
                    val_predictions.extend(predictions)
                    for i in range(len(img_metas)):
                        target = {
                            'gt_bboxes_3d': gt_bboxes_3d[i],
                            'gt_labels_3d': gt_labels_3d[i],
                            'img_metas': img_metas[i]
                        }
                        if gt_masks is not None:
                            target['gt_masks'] = gt_masks[i]
                        val_targets.append(target)
            
            avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
            val_losses.append(avg_val_loss)
            
            # Evaluate on validation set
            if val_predictions:
                epoch_metrics = self.evaluator.evaluate(val_predictions, val_targets)
                val_metrics.append(epoch_metrics)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}")
        
        # Save training results
        training_results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics
        }
        
        # Save model checkpoint
        checkpoint_dir = self.demo_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_results': training_results,
            'args': self.args
        }, checkpoint_dir / 'demo_model.pth')
        
        print(f"✅ Training completed! Model saved to {checkpoint_dir / 'demo_model.pth'}")
        
        # Visualize training progress
        self.visualize_training_progress(training_results)
        
        return training_results
    
    def visualize_training_progress(self, training_results):
        """Visualize training progress."""
        print("📊 Creating training visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(training_results['train_losses']) + 1)
        axes[0, 0].plot(epochs, training_results['train_losses'], 'b-', label='Train')
        axes[0, 0].plot(epochs, training_results['val_losses'], 'r-', label='Validation')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation metrics
        if training_results['val_metrics']:
            overall_scores = [m.get('overall_score', 0) for m in training_results['val_metrics']]
            map_scores = [m.get('3d_mAP_3d', 0) for m in training_results['val_metrics']]
            
            axes[0, 1].plot(epochs, overall_scores, 'g-', label='Overall Score')
            axes[0, 1].set_title('Overall Performance')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].plot(epochs, map_scores, 'm-', label='3D mAP')
            axes[1, 0].set_title('3D Detection mAP')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Model architecture summary
        axes[1, 1].text(0.1, 0.9, "Model Architecture:", fontsize=12, weight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.8, f"Fusion Mode: {self.args.fusion_mode}", fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.7, f"Device: {self.device}", fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Training Epochs: {self.args.epochs}", fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.5, f"Final Train Loss: {training_results['train_losses'][-1]:.4f}", fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f"Final Val Loss: {training_results['val_losses'][-1]:.4f}", fontsize=10, transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.demo_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Training progress saved to {self.demo_dir / 'training_progress.png'}")
    
    def run_demo_testing(self, model, test_loader):
        """Run demonstration testing."""
        print("\n🧪 Running demo testing...")
        
        model.eval()
        
        all_predictions = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 5:  # Limit for demo
                    break
                    
                # Move data to device
                images = batch['images'].to(self.device)
                img_metas = batch['img_metas']
                
                # Measure inference time
                start_time = time.time()
                predictions = model.forward_test(images, img_metas)
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000)  # ms
                all_predictions.extend(predictions)
                
                # Create targets
                for i in range(len(img_metas)):
                    target = {
                        'gt_bboxes_3d': batch['gt_bboxes_3d'][i],
                        'gt_labels_3d': batch['gt_labels_3d'][i],
                        'img_metas': img_metas[i]
                    }
                    if batch.get('gt_masks') is not None:
                        target['gt_masks'] = batch['gt_masks'][i]
                    all_targets.append(target)
                
                # Visualize sample
                if batch_idx < 3:  # Visualize first 3 samples
                    self.visualize_test_sample(batch, predictions, batch_idx)
        
        # Evaluate results
        test_metrics = self.evaluator.evaluate(all_predictions, all_targets)
        
        # Add performance metrics
        test_metrics.update({
            'avg_inference_time_ms': np.mean(inference_times),
            'fps': 1000 / np.mean(inference_times),
            'total_samples': len(all_predictions)
        })
        
        # Save test results
        with open(self.demo_dir / 'test_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_metrics = {}
            for key, value in test_metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = value.item()
                else:
                    serializable_metrics[key] = value
            json.dump(serializable_metrics, f, indent=2)
        
        print("✅ Testing completed!")
        print(f"   Test samples: {len(all_predictions)}")
        print(f"   Average inference time: {np.mean(inference_times):.2f} ms")
        print(f"   FPS: {1000 / np.mean(inference_times):.2f}")
        print(f"   Overall score: {test_metrics.get('overall_score', 0):.4f}")
        
        return test_metrics
    
    def visualize_test_sample(self, batch, predictions, batch_idx):
        """Visualize a test sample with predictions."""
        
        save_path = self.demo_dir / 'visualizations' / f'test_sample_{batch_idx}.png'
        
        self.visualizer.visualize_sample(
            images=batch['images'][0],
            predictions=predictions[0],
            targets={
                'gt_bboxes_3d': batch['gt_bboxes_3d'][0],
                'gt_labels_3d': batch['gt_labels_3d'][0],
                'gt_masks': batch.get('gt_masks', [None])[0]
            },
            img_meta=batch['img_metas'][0],
            save_path=save_path
        )
    
    def generate_demo_report(self, training_results=None, test_results=None):
        """Generate comprehensive demo report."""
        print("\n📄 Generating demo report...")
        
        report_lines = [
            "BEVNeXt-SAM2 Fusion Model - Demo Report",
            "=" * 50,
            "",
            f"Demo Configuration:",
            f"  - Mode: {self.args.mode}",
            f"  - Fusion Mode: {self.args.fusion_mode}",
            f"  - Device: {self.device}",
            f"  - Samples: {self.args.num_samples}",
            ""
        ]
        
        if training_results:
            final_train_loss = training_results['train_losses'][-1]
            final_val_loss = training_results['val_losses'][-1]
            
            report_lines.extend([
                "Training Results:",
                f"  - Epochs: {len(training_results['train_losses'])}",
                f"  - Final training loss: {final_train_loss:.4f}",
                f"  - Final validation loss: {final_val_loss:.4f}",
                ""
            ])
        
        if test_results:
            report_lines.extend([
                "Testing Results:",
                f"  - Test samples: {test_results.get('total_samples', 'N/A')}",
                f"  - Average inference time: {test_results.get('avg_inference_time_ms', 0):.2f} ms",
                f"  - FPS: {test_results.get('fps', 0):.2f}",
                f"  - Overall score: {test_results.get('overall_score', 0):.4f}",
                f"  - 3D mAP: {test_results.get('3d_mAP_3d', 0):.4f}",
                f"  - 2D IoU: {test_results.get('2d_IoU', 0):.4f}",
                ""
            ])
        
        report_lines.extend([
            "Generated Files:",
            f"  - Model checkpoint: checkpoints/demo_model.pth",
            f"  - Training progress: training_progress.png",
            f"  - Test visualizations: visualizations/",
            f"  - Test results: test_results.json",
            f"  - This report: demo_report.txt",
            ""
        ])
        
        # Save report
        with open(self.demo_dir / 'demo_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Demo report saved to {self.demo_dir / 'demo_report.txt'}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 50)
        print(f"Demo directory: {self.demo_dir}")
        print("Key files generated:")
        print(f"  📊 Training progress: training_progress.png")
        print(f"  🎯 Model checkpoint: checkpoints/demo_model.pth")
        print(f"  🖼️  Test visualizations: visualizations/")
        print(f"  📋 Demo report: demo_report.txt")
        
    def run_training_demo(self):
        """Run complete training demonstration."""
        configs = self.create_demo_configs()
        model = self.setup_demo_model(configs)
        data_loaders = self.setup_demo_data(configs['data_config'])
        
        training_results = self.run_demo_training(model, data_loaders)
        test_results = self.run_demo_testing(model, data_loaders['test_loader'])
        
        self.generate_demo_report(training_results, test_results)
        
    def run_testing_demo(self):
        """Run testing-only demonstration."""
        if not self.args.checkpoint:
            print("❌ Checkpoint required for testing mode!")
            return
            
        configs = self.create_demo_configs()
        model = self.setup_demo_model(configs)
        
        # Load checkpoint
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        data_loaders = self.setup_demo_data(configs['data_config'])
        test_results = self.run_demo_testing(model, data_loaders['test_loader'])
        
        self.generate_demo_report(test_results=test_results)
        
    def run_inference_demo(self):
        """Run simple inference demonstration."""
        configs = self.create_demo_configs()
        model = self.setup_demo_model(configs)
        
        if self.args.checkpoint:
            checkpoint = torch.load(self.args.checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create single sample
        dataset = SyntheticFusionDataset(num_samples=1, load_masks=True)
        sample = dataset[0]
        
        # Run inference
        model.eval()
        with torch.no_grad():
            images = sample['images'].unsqueeze(0).to(self.device)
            img_metas = [sample['img_metas']]
            
            start_time = time.time()
            predictions = model.forward_test(images, img_metas)
            end_time = time.time()
        
        # Visualize result
        self.visualizer.visualize_sample(
            images=sample['images'],
            predictions=predictions[0],
            targets={
                'gt_bboxes_3d': sample['gt_bboxes_3d'],
                'gt_labels_3d': sample['gt_labels_3d'],
                'gt_masks': sample.get('gt_masks')
            },
            img_meta=sample['img_metas'],
            save_path=self.demo_dir / 'inference_demo.png'
        )
        
        print(f"✅ Inference demo completed!")
        print(f"   Inference time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"   Result saved to: {self.demo_dir / 'inference_demo.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description='BEVNeXt-SAM2 Fusion Demo')
    
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['train', 'test', 'demo'],
                       help='Demo mode')
    parser.add_argument('--demo-dir', type=str, default='demo_results',
                       help='Demo output directory')
    parser.add_argument('--checkpoint', type=str,
                       help='Model checkpoint for testing')
    parser.add_argument('--sam2-checkpoint', type=str,
                       help='SAM2 checkpoint file')
    
    # Model configuration
    parser.add_argument('--fusion-mode', type=str, default='feature_fusion',
                       choices=['feature_fusion', 'late_fusion', 'multi_scale'],
                       help='Fusion mode')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of synthetic samples')
    
    # Options
    parser.add_argument('--use-synthetic', action='store_true', default=True,
                       help='Use synthetic data')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create demo instance
    demo = FusionDemo(args)
    
    # Run appropriate demo mode
    if args.mode == 'train':
        demo.run_training_demo()
    elif args.mode == 'test':
        demo.run_testing_demo()
    else:  # demo mode
        demo.run_inference_demo()


if __name__ == '__main__':
    main()