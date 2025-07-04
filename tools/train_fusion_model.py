#!/usr/bin/env python3
"""
BEVNeXt-SAM2 Fusion Model Training Script

Comprehensive training pipeline for the fusion model combining BEVNeXt's 3D detection
with SAM2's segmentation capabilities.

Features:
- Multi-task training (3D detection + 2D segmentation)
- Validation during training
- Advanced visualization
- Checkpoint management
- TensorBoard logging
- Mixed precision training
- Learning rate scheduling
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.bevnext_sam2_fusion_model import BEVNeXtSAM2FusionModel
from tools.fusion_dataset import FusionDataset, create_fusion_dataloader
from tools.fusion_evaluator import FusionEvaluator
from tools.fusion_visualizer import FusionVisualizer
from tools.lr_scheduler import CosineLRScheduler


class FusionTrainer:
    """
    Comprehensive trainer for BEVNeXt-SAM2 fusion model.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_metric = 0.0
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Build model
        self.build_model()
        
        # Setup data
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup evaluation and visualization
        self.setup_evaluation()
        
        # Mixed precision training
        self.scaler = GradScaler() if args.use_amp else None
        
        print(f"🚀 Training setup complete!")
        print(f"   Device: {self.device}")
        print(f"   Model: {args.fusion_mode} fusion")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.lr}")
        
    def setup_directories(self):
        """Create necessary directories for training."""
        self.work_dir = Path(self.args.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.work_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.work_dir / 'logs').mkdir(exist_ok=True)
        (self.work_dir / 'visualizations').mkdir(exist_ok=True)
        (self.work_dir / 'tensorboard').mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup TensorBoard and file logging."""
        self.writer = SummaryWriter(self.work_dir / 'tensorboard')
        
        # Setup file logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.work_dir / 'logs' / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def build_model(self):
        """Build the fusion model."""
        print("🔧 Building fusion model...")
        
        self.model = BEVNeXtSAM2FusionModel(
            bev_config=self.args.bev_config,
            sam2_config=self.args.sam2_config,
            sam2_checkpoint=self.args.sam2_checkpoint,
            fusion_mode=self.args.fusion_mode,
            bev_weight=self.args.bev_weight,
            sam_weight=self.args.sam_weight,
            fusion_weight=self.args.fusion_weight,
            device=self.device
        ).to(self.device)
        
        # Load checkpoint if specified
        if self.args.resume_from:
            self.load_checkpoint(self.args.resume_from)
            
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
    def setup_data(self):
        """Setup data loaders for training and validation."""
        print("📊 Setting up datasets...")
        
        # Training dataset
        self.train_dataset = FusionDataset(
            data_config=self.args.data_config,
            split='train',
            transform=self.get_train_transforms()
        )
        
        self.train_loader = create_fusion_dataloader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # Validation dataset
        self.val_dataset = FusionDataset(
            data_config=self.args.data_config,
            split='val',
            transform=self.get_val_transforms()
        )
        
        self.val_loader = create_fusion_dataloader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"   Training samples: {len(self.train_dataset)}")
        print(f"   Validation samples: {len(self.val_dataset)}")
        
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        
        # Separate parameters for different components
        bev_params = []
        sam_params = []
        fusion_params = []
        
        for name, param in self.model.named_parameters():
            if 'bev_detector' in name:
                bev_params.append(param)
            elif 'sam2_model' in name:
                sam_params.append(param)
                param.requires_grad = self.args.train_sam2  # Control SAM2 training
            else:
                fusion_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': bev_params, 'lr': self.args.lr},
            {'params': sam_params, 'lr': self.args.lr * 0.1},  # Lower LR for pre-trained SAM2
            {'params': fusion_params, 'lr': self.args.lr}
        ]
        
        # Optimizer
        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(param_groups, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
        
        # Learning rate scheduler
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.args.epochs,
            lr_min=self.args.lr * 0.01,
            warmup_t=self.args.warmup_epochs,
            warmup_lr_init=self.args.lr * 0.1
        )
        
    def setup_evaluation(self):
        """Setup evaluation and visualization tools."""
        self.evaluator = FusionEvaluator(self.args.eval_config)
        self.visualizer = FusionVisualizer(self.work_dir / 'visualizations')
        
    def get_train_transforms(self):
        """Get training data transforms."""
        # Implement data augmentation for training
        transforms = [
            # Add your transforms here
        ]
        return transforms
        
    def get_val_transforms(self):
        """Get validation data transforms."""
        # Implement validation transforms (usually minimal)
        transforms = [
            # Add your transforms here
        ]
        return transforms
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.args.epochs}',
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['images'].to(self.device)
            img_metas = batch['img_metas']
            gt_bboxes_3d = [bbox.to(self.device) for bbox in batch['gt_bboxes_3d']]
            gt_labels_3d = [labels.to(self.device) for labels in batch['gt_labels_3d']]
            gt_masks = batch.get('gt_masks')
            if gt_masks is not None:
                gt_masks = [mask.to(self.device) for mask in gt_masks]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.args.use_amp:
                with autocast():
                    losses = self.model.forward_train(
                        images=images,
                        img_metas=img_metas,
                        gt_bboxes_3d=gt_bboxes_3d,
                        gt_labels_3d=gt_labels_3d,
                        gt_masks=gt_masks
                    )
                
                # Backward pass
                self.scaler.scale(losses['total']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model.forward_train(
                    images=images,
                    img_metas=img_metas,
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    gt_masks=gt_masks
                )
                
                # Backward pass
                losses['total'].backward()
                self.optimizer.step()
            
            # Update metrics
            batch_loss = losses['total'].item()
            total_loss += batch_loss
            
            # Accumulate loss components
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log batch metrics
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', batch_loss, global_step)
            
            # Visualize periodically
            if batch_idx % self.args.vis_interval == 0:
                self.visualize_batch(batch, epoch, batch_idx, 'train')
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        loss_components['total'] = avg_loss
        return loss_components
        
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f'Validation Epoch {epoch}',
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                images = batch['images'].to(self.device)
                img_metas = batch['img_metas']
                gt_bboxes_3d = [bbox.to(self.device) for bbox in batch['gt_bboxes_3d']]
                gt_labels_3d = [labels.to(self.device) for labels in batch['gt_labels_3d']]
                gt_masks = batch.get('gt_masks')
                if gt_masks is not None:
                    gt_masks = [mask.to(self.device) for mask in gt_masks]
                
                # Forward pass
                predictions = self.model.forward_test(images, img_metas)
                
                # Compute validation loss
                losses = self.model.forward_train(
                    images=images,
                    img_metas=img_metas,
                    gt_bboxes_3d=gt_bboxes_3d,
                    gt_labels_3d=gt_labels_3d,
                    gt_masks=gt_masks
                )
                
                # Accumulate metrics
                batch_loss = losses['total'].item()
                total_loss += batch_loss
                
                for key, value in losses.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
                
                # Store predictions and targets for evaluation
                all_predictions.extend(predictions)
                all_targets.extend([{
                    'gt_bboxes_3d': gt_bboxes_3d[i],
                    'gt_labels_3d': gt_labels_3d[i],
                    'gt_masks': gt_masks[i] if gt_masks else None,
                    'img_metas': img_metas[i]
                } for i in range(len(img_metas))])
                
                # Update progress bar
                progress_bar.set_postfix({'val_loss': f'{batch_loss:.4f}'})
                
                # Visualize periodically
                if batch_idx % self.args.vis_interval == 0:
                    self.visualize_batch(batch, epoch, batch_idx, 'val')
        
        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        # Comprehensive evaluation
        eval_metrics = self.evaluator.evaluate(all_predictions, all_targets)
        
        # Combine loss and evaluation metrics
        val_metrics = {**loss_components, **eval_metrics}
        val_metrics['total'] = avg_loss
        
        return val_metrics
        
    def train(self):
        """Main training loop."""
        print("\n🎯 Starting training...")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step(epoch)
            
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            self.log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Save checkpoint
            is_best = val_metrics.get('mAP_3d', 0.0) > self.best_metric
            if is_best:
                self.best_metric = val_metrics.get('mAP_3d', 0.0)
            
            self.save_checkpoint(epoch, is_best)
            
            # Generate epoch visualization
            self.visualize_epoch_summary(epoch, train_metrics, val_metrics)
            
            print(f"Epoch {epoch}: train_loss={train_metrics['total']:.4f}, "
                  f"val_loss={val_metrics['total']:.4f}, "
                  f"val_mAP={val_metrics.get('mAP_3d', 0.0):.4f}, "
                  f"time={epoch_time:.1f}s")
        
        print("✅ Training completed!")
        self.writer.close()
        
    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log metrics for the epoch."""
        
        # TensorBoard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        self.writer.add_scalar('train/epoch_time', epoch_time, epoch)
        self.writer.add_scalar('train/learning_rate', 
                              self.optimizer.param_groups[0]['lr'], epoch)
        
        # File logging
        self.logger.info(f"Epoch {epoch}: "
                        f"train_loss={train_metrics['total']:.4f}, "
                        f"val_loss={val_metrics['total']:.4f}, "
                        f"time={epoch_time:.1f}s")
        
    def visualize_batch(self, batch: Dict, epoch: int, batch_idx: int, split: str):
        """Visualize a training/validation batch."""
        if not hasattr(self, 'vis_count'):
            self.vis_count = 0
        
        self.vis_count += 1
        if self.vis_count % 10 != 0:  # Only visualize every 10th call
            return
        
        # Get model predictions
        with torch.no_grad():
            self.model.eval()
            predictions = self.model.forward_test(
                batch['images'].to(self.device),
                batch['img_metas']
            )
            self.model.train()
        
        # Visualize first sample in batch
        self.visualizer.visualize_sample(
            images=batch['images'][0],
            predictions=predictions[0],
            targets={
                'gt_bboxes_3d': batch['gt_bboxes_3d'][0],
                'gt_labels_3d': batch['gt_labels_3d'][0],
                'gt_masks': batch.get('gt_masks', [None])[0]
            },
            img_meta=batch['img_metas'][0],
            save_path=self.work_dir / 'visualizations' / f'{split}_epoch_{epoch}_batch_{batch_idx}.png'
        )
        
    def visualize_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Create epoch summary visualization."""
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves (you'll need to store history for this)
        # axes[0, 0].plot(train_loss_history, label='Train')
        # axes[0, 0].plot(val_loss_history, label='Val')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        
        # mAP curves
        # axes[0, 1].plot(val_mAP_history)
        axes[0, 1].set_title('Validation mAP')
        
        # Learning rate
        # axes[1, 0].plot(lr_history)
        axes[1, 0].set_title('Learning Rate')
        
        # Other metrics
        axes[1, 1].text(0.1, 0.5, f"Epoch: {epoch}\nTrain Loss: {train_metrics['total']:.4f}\n"
                                   f"Val Loss: {val_metrics['total']:.4f}\n"
                                   f"Val mAP: {val_metrics.get('mAP_3d', 0.0):.4f}")
        axes[1, 1].set_title('Current Metrics')
        
        plt.tight_layout()
        plt.savefig(self.work_dir / 'visualizations' / f'epoch_{epoch}_summary.png')
        plt.close()
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'args': self.args
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.work_dir / 'checkpoints' / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.work_dir / 'checkpoints' / 'best.pth')
        
        # Save periodic checkpoints
        if epoch % self.args.save_interval == 0:
            torch.save(checkpoint, 
                      self.work_dir / 'checkpoints' / f'epoch_{epoch}.pth')
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        print(f"Resumed from epoch {self.start_epoch}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train BEVNeXt-SAM2 Fusion Model')
    
    # Model config
    parser.add_argument('--bev-config', type=str, required=True,
                       help='BEVNeXt configuration file')
    parser.add_argument('--sam2-config', type=str, required=True,
                       help='SAM2 configuration file')
    parser.add_argument('--sam2-checkpoint', type=str,
                       help='Pre-trained SAM2 checkpoint')
    parser.add_argument('--fusion-mode', type=str, default='feature_fusion',
                       choices=['feature_fusion', 'late_fusion', 'multi_scale'],
                       help='Fusion strategy')
    
    # Training config
    parser.add_argument('--data-config', type=str, required=True,
                       help='Dataset configuration file')
    parser.add_argument('--work-dir', type=str, default='work_dirs/fusion_training',
                       help='Working directory for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs')
    
    # Loss weights
    parser.add_argument('--bev-weight', type=float, default=1.0,
                       help='Weight for BEV detection loss')
    parser.add_argument('--sam-weight', type=float, default=1.0,
                       help='Weight for SAM segmentation loss')
    parser.add_argument('--fusion-weight', type=float, default=0.5,
                       help='Weight for fusion consistency loss')
    
    # Training options
    parser.add_argument('--train-sam2', action='store_true',
                       help='Whether to fine-tune SAM2 parameters')
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--resume-from', type=str,
                       help='Resume training from checkpoint')
    
    # Logging and saving
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--vis-interval', type=int, default=100,
                       help='Visualize every N batches')
    parser.add_argument('--eval-config', type=str,
                       help='Evaluation configuration file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create trainer and start training
    trainer = FusionTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()