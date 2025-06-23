#!/usr/bin/env python3
"""
BEVNeXt-SAM2 Training Script
Comprehensive training pipeline for 3D object detection with SAM2 integration
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, '/workspace/bevnext-sam2')

class BEVNeXtSAM2Model(nn.Module):
    """Complete BEVNeXt-SAM2 model for training"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.get('gradient_checkpointing', False)
        
        # Camera feature backbone
        self.camera_backbone = self._build_camera_backbone()
        
        # BEV transformer
        self.bev_transformer = self._build_bev_transformer()
        
        # SAM2 integration
        self.sam2_fusion = self._build_sam2_fusion()
        
        # Detection heads
        self.detection_head = self._build_detection_head()
        
        # Loss functions
        self.setup_losses()
        
        # Enable gradient checkpointing for memory efficiency if configured
        if self.use_gradient_checkpointing:
            print("✓ Gradient checkpointing enabled for memory efficiency")
            if hasattr(self.bev_transformer, 'layers'):
                for layer in self.bev_transformer.layers:
                    layer.use_checkpoint = True
        
    def _build_camera_backbone(self):
        """Build camera feature extraction backbone"""
        layers = []
        
        # Simple CNN backbone for camera features
        in_channels = 3
        for out_channels in [64, 128, 256, 512]:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def _build_bev_transformer(self):
        """Build BEV transformer for spatial reasoning"""
        d_model = self.config['d_model']
        nhead = self.config['nhead']
        num_layers = self.config['num_transformer_layers']
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def _build_sam2_fusion(self):
        """Build SAM2 fusion module"""
        d_model = self.config['d_model']
        return nn.Sequential(
            nn.Conv2d(d_model + d_model, d_model, 3, padding=1),  # BEV features + SAM2 features
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
    
    def _build_detection_head(self):
        """Build 3D object detection head"""
        num_classes = self.config['num_classes']
        d_model = self.config['d_model']
        
        return nn.ModuleDict({
            'classification': nn.Linear(d_model, num_classes),
            'regression': nn.Linear(d_model, 10),  # x,y,z,w,l,h,rot,vx,vy,vz
            'confidence': nn.Linear(d_model, 1)
        })
    
    def setup_losses(self):
        """Setup loss functions"""
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()
        self.conf_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass through the complete model"""
        # Extract inputs
        camera_images = batch['camera_images']  # [B, N_cams, 3, H, W]
        sam2_masks = batch['sam2_masks']        # [B, N_cams, 1, H, W]
        
        B, N_cams = camera_images.shape[:2]
        
        # Process camera features with checkpointing
        camera_features = []
        for cam_idx in range(N_cams):
            cam_img = camera_images[:, cam_idx]  # [B, 3, H, W]
            if self.training and self.use_gradient_checkpointing:
                cam_feat = checkpoint(self.camera_backbone, cam_img, use_reentrant=False)
            else:
                cam_feat = self.camera_backbone(cam_img)
            camera_features.append(cam_feat)
            # Clear intermediate results
            del cam_img, cam_feat
        
        # Stack camera features
        camera_features = torch.stack(camera_features, dim=1)  # [B, N_cams, 512, H', W']
        
        # Project to BEV space (simplified projection)
        bev_features = self._project_to_bev(camera_features)  # [B, d_model, BEV_H, BEV_W]
        del camera_features  # Free memory
        
        # Integrate SAM2 features
        sam2_features = self._process_sam2_masks(sam2_masks)  # [B, d_model, BEV_H, BEV_W]
        fused_features = self.sam2_fusion(torch.cat([bev_features, sam2_features], dim=1))
        del bev_features, sam2_features  # Free memory
        
        # Apply BEV transformer
        B, C, H, W = fused_features.shape
        flat_features = fused_features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        del fused_features  # Free memory
        
        # Add positional encoding
        pos_encoding = self._get_positional_encoding(H, W, C).to(flat_features.device)
        flat_features = flat_features + pos_encoding.unsqueeze(0)
        del pos_encoding  # Free memory
        
        # Transform features with checkpointing
        if self.training and self.use_gradient_checkpointing:
            transformed_features = checkpoint(self.bev_transformer, flat_features, use_reentrant=False)
        else:
            transformed_features = self.bev_transformer(flat_features)
        del flat_features  # Free memory
        
        # Object queries (simplified)
        num_queries = self.config['num_queries']
        object_queries = transformed_features[:, :num_queries]  # [B, num_queries, C]
        del transformed_features  # Free memory
        
        # Predictions
        predictions = {
            'classification': self.detection_head['classification'](object_queries),
            'regression': self.detection_head['regression'](object_queries),
            'confidence': self.detection_head['confidence'](object_queries)
        }
        
        return predictions
    
    def _project_to_bev(self, camera_features: torch.Tensor) -> torch.Tensor:
        """Project camera features to BEV space"""
        B, N_cams, C, H, W = camera_features.shape
        bev_h, bev_w = self.config['bev_size']
        d_model = self.config['d_model']
        
        # Simplified projection - average pool across cameras
        pooled = camera_features.mean(dim=1)  # [B, C, H, W]
        
        # Resize to BEV dimensions
        bev_features = F.interpolate(pooled, size=(bev_h, bev_w), mode='bilinear', align_corners=False)
        
        # Project to desired channel dimension (use d_model instead of hardcoded 256)
        bev_features = F.conv2d(bev_features, 
                               weight=torch.randn(d_model, C, 1, 1).to(camera_features.device),
                               bias=None)
        
        return bev_features
    
    def _process_sam2_masks(self, sam2_masks: torch.Tensor) -> torch.Tensor:
        """Process SAM2 masks to BEV features"""
        B, N_cams, _, H, W = sam2_masks.shape
        bev_h, bev_w = self.config['bev_size']
        d_model = self.config['d_model']
        
        # Average across cameras and resize
        pooled_masks = sam2_masks.mean(dim=1)  # [B, 1, H, W]
        bev_masks = F.interpolate(pooled_masks, size=(bev_h, bev_w), mode='bilinear', align_corners=False)
        
        # Expand to feature dimensions (use d_model instead of hardcoded 256)
        bev_sam2_features = bev_masks.repeat(1, d_model, 1, 1)
        
        return bev_sam2_features
    
    def _get_positional_encoding(self, H: int, W: int, C: int) -> torch.Tensor:
        """Generate positional encoding for BEV features"""
        pe = torch.zeros(H * W, C)
        position = torch.arange(0, H * W).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, C, 2).float() * 
                           -(math.log(10000.0) / C))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def compute_loss(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute training losses"""
        losses = {}
        
        # Classification loss
        pred_cls = predictions['classification']  # [B, num_queries, num_classes]
        target_cls = targets['labels']            # [B, num_queries]
        
        losses['classification'] = self.cls_loss(
            pred_cls.reshape(-1, pred_cls.size(-1)),
            target_cls.reshape(-1)
        )
        
        # Regression loss (only for positive samples)
        pred_reg = predictions['regression']      # [B, num_queries, 10]
        target_reg = targets['boxes']            # [B, num_queries, 10]
        target_valid = targets['valid_mask']     # [B, num_queries]
        
        if target_valid.sum() > 0:
            losses['regression'] = self.reg_loss(
                pred_reg[target_valid],
                target_reg[target_valid]
            )
        else:
            losses['regression'] = torch.tensor(0.0, device=pred_reg.device)
        
        # Confidence loss
        pred_conf = predictions['confidence']     # [B, num_queries, 1]
        target_conf = target_valid.float().unsqueeze(-1)  # [B, num_queries, 1]
        
        losses['confidence'] = self.conf_loss(pred_conf, target_conf)
        
        # Total loss
        losses['total'] = (losses['classification'] + 
                          losses['regression'] + 
                          losses['confidence'])
        
        return losses


class SyntheticDataset(Dataset):
    """Synthetic dataset for training demonstration"""
    
    def __init__(self, config: Dict, split: str = 'train'):
        self.config = config
        self.split = split
        self.num_samples = config['num_samples'][split]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic data
        B = 1
        N_cams = self.config['num_cameras']
        H, W = self.config['image_size']
        bev_h, bev_w = self.config['bev_size']
        num_queries = self.config['num_queries']
        
        # Synthetic camera images
        camera_images = torch.randn(B, N_cams, 3, H, W)
        
        # Synthetic SAM2 masks
        sam2_masks = torch.randint(0, 2, (B, N_cams, 1, H, W)).float()
        
        # Synthetic targets
        labels = torch.randint(0, self.config['num_classes'], (B, num_queries))
        boxes = torch.randn(B, num_queries, 10)
        valid_mask = torch.randint(0, 2, (B, num_queries)).bool()
        
        return {
            'camera_images': camera_images.squeeze(0),
            'sam2_masks': sam2_masks.squeeze(0),
            'labels': labels.squeeze(0),
            'boxes': boxes.squeeze(0),
            'valid_mask': valid_mask.squeeze(0)
        }


class Trainer:
    """Main training class"""
    
    def __init__(self, config: Dict, use_mixed_precision: bool = False):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        
        # Setup model
        self.model = BEVNeXtSAM2Model(config).to(self.device)
        
        # Enable model optimizations for GPU
        if torch.cuda.is_available():
            # Compile model for faster training (PyTorch 2.0+)
            try:
                self.model = torch.compile(self.model)
                print("✓ Model compiled for faster training")
            except:
                print("⚠ Model compilation not available, continuing without")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✓ Mixed precision training enabled")
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
        
        # Setup data loaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create data loader for given split"""
        dataset = SyntheticDataset(self.config, split)
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=(split == 'train'),
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
    
    def setup_logging(self):
        """Setup logging and tensorboard"""
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} Train')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(batch)
                    
                    # Compute losses
                    targets = {
                        'labels': batch['labels'],
                        'boxes': batch['boxes'],
                        'valid_mask': batch['valid_mask']
                    }
                    losses = self.model.compute_loss(predictions, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(losses['total']).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(batch)
                
                # Compute losses
                targets = {
                    'labels': batch['labels'],
                    'boxes': batch['boxes'],
                    'valid_mask': batch['valid_mask']
                }
                losses = self.model.compute_loss(predictions, targets)
                
                # Backward pass
                losses['total'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total'].item()
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'cls': f"{losses['classification'].item():.4f}",
                'reg': f"{losses['regression'].item():.4f}",
                'conf': f"{losses['confidence'].item():.4f}"
            })
            
            # Log to tensorboard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss_total', losses['total'].item(), global_step)
            self.writer.add_scalar('train/loss_cls', losses['classification'].item(), global_step)
            self.writer.add_scalar('train/loss_reg', losses['regression'].item(), global_step)
            self.writer.add_scalar('train/loss_conf', losses['confidence'].item(), global_step)
            
            # Clear GPU cache periodically to prevent memory accumulation
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'avg_loss': avg_loss, **loss_components}
    
    def validate(self) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        loss_components = {}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch+1} Val')
            
            for batch in pbar:
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass with mixed precision
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(batch)
                        
                        # Compute losses
                        targets = {
                            'labels': batch['labels'],
                            'boxes': batch['boxes'], 
                            'valid_mask': batch['valid_mask']
                        }
                        losses = self.model.compute_loss(predictions, targets)
                else:
                    predictions = self.model(batch)
                    
                    # Compute losses
                    targets = {
                        'labels': batch['labels'],
                        'boxes': batch['boxes'], 
                        'valid_mask': batch['valid_mask']
                    }
                    losses = self.model.compute_loss(predictions, targets)
                
                # Update metrics
                total_loss += losses['total'].item()
                for key, value in losses.items():
                    if key not in loss_components:
                        loss_components[key] = 0
                    loss_components[key] += value.item()
                
                pbar.set_postfix({'val_loss': f"{losses['total'].item():.4f}"})
                
                # Clear GPU cache periodically during validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Average losses
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {'avg_loss': avg_loss, **loss_components}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting BEVNeXt-SAM2 training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_metrics['avg_loss']:.4f} - "
                f"Val Loss: {val_metrics['avg_loss']:.4f} - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_metrics['avg_loss'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['avg_loss'], epoch)
            self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['avg_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['avg_loss']
            
            self.save_checkpoint(is_best)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                epoch_checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config
                }, epoch_checkpoint_path)
        
        self.logger.info("Training completed!")
        self.writer.close()


def get_gpu_memory_config():
    """Get GPU memory-optimized configuration based on available VRAM"""
    if not torch.cuda.is_available():
        return get_cpu_config()
    
    # Get GPU memory in GB
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🔍 Detected GPU memory: {gpu_memory_gb:.1f} GB")
    
    if gpu_memory_gb >= 20:
        # High-end GPU (>= 20GB) - Full capacity
        print("🚀 Using HIGH-END GPU configuration (>=20GB VRAM)")
        return {
            'd_model': 512,
            'nhead': 16,
            'num_transformer_layers': 8,
            'num_classes': 10,
            'num_queries': 1000,
            'bev_size': [128, 128],
            'image_size': [224, 224],
            'num_cameras': 6,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 50,
            'num_workers': 8,
            'num_samples': {'train': 5000, 'val': 1000},
            'memory_fraction': 0.9,
            'gradient_checkpointing': False,
            'output_dir': '/workspace/outputs/training_gpu_high'
        }
    elif gpu_memory_gb >= 12:
        # Mid-range GPU (12-20GB) - Balanced
        print("⚡ Using MID-RANGE GPU configuration (12-20GB VRAM)")
        return {
            'd_model': 256,
            'nhead': 8,
            'num_transformer_layers': 6,
            'num_classes': 10,
            'num_queries': 600,
            'bev_size': [64, 64],
            'image_size': [224, 224],
            'num_cameras': 6,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 50,
            'num_workers': 6,
            'num_samples': {'train': 2000, 'val': 400},
            'memory_fraction': 0.85,
            'gradient_checkpointing': True,
            'output_dir': '/workspace/outputs/training_gpu_mid'
        }
    elif gpu_memory_gb >= 6:
        # Low-end GPU (6-12GB) - Conservative
        print("💡 Using LOW-END GPU configuration (6-12GB VRAM)")
        return {
            'd_model': 128,
            'nhead': 8,
            'num_transformer_layers': 4,
            'num_classes': 10,
            'num_queries': 200,
            'bev_size': [48, 48],
            'image_size': [224, 224],
            'num_cameras': 6,
            'batch_size': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 50,
            'num_workers': 4,
            'num_samples': {'train': 1000, 'val': 200},
            'memory_fraction': 0.7,
            'gradient_checkpointing': True,
            'output_dir': '/workspace/outputs/training_gpu_low'
        }
    else:
        # Ultra low-end GPU (<6GB) - Minimal
        print("🔋 Using ULTRA-LOW GPU configuration (<6GB VRAM)")
        return {
            'd_model': 64,
            'nhead': 4,
            'num_transformer_layers': 2,
            'num_classes': 10,
            'num_queries': 100,
            'bev_size': [32, 32],
            'image_size': [224, 224],
            'num_cameras': 6,
            'batch_size': 1,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_epochs': 50,
            'num_workers': 2,
            'num_samples': {'train': 500, 'val': 100},
            'memory_fraction': 0.6,
            'gradient_checkpointing': True,
            'output_dir': '/workspace/outputs/training_gpu_ultra'
        }

def get_cpu_config():
    """Get CPU-optimized configuration"""
    print("🖥️  Using CPU configuration")
    return {
        'd_model': 128,
        'nhead': 4,
        'num_transformer_layers': 3,
        'num_classes': 10,
        'num_queries': 300,
        'bev_size': [32, 32],
        'image_size': [224, 224],
        'num_cameras': 6,
        'batch_size': 1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 10,
        'num_workers': 0,
        'num_samples': {'train': 100, 'val': 20},
        'memory_fraction': 1.0,
        'gradient_checkpointing': False,
        'output_dir': '/workspace/outputs/training_cpu'
    }

def apply_memory_optimizations(config):
    """Apply GPU memory optimizations based on configuration"""
    if torch.cuda.is_available():
        memory_fraction = config.get('memory_fraction', 0.8)
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        print(f"✓ GPU memory fraction set to {memory_fraction}")
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("✓ Flash attention enabled for memory efficiency")
        except:
            print("⚠ Flash attention not available, using standard attention")
        
        # Set memory allocation strategy based on GPU tier (more conservative)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 12:
            split_size = 256
        elif gpu_memory_gb >= 8:
            split_size = 64  # More conservative for 8GB GPUs
        elif gpu_memory_gb >= 6:
            split_size = 32  # Very conservative for 6-8GB GPUs  
        else:
            split_size = 16  # Ultra conservative for <6GB GPUs
            
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{split_size}'
        print(f"✓ Memory split size set to {split_size}MB")
        
        # Additional memory optimizations for low-end GPUs
        if gpu_memory_gb < 8:
            # Enable memory efficient mode
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] += ',garbage_collection_threshold:0.6'
            print("✓ Aggressive garbage collection enabled for low VRAM")
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")

def get_default_config():
    """Get default training configuration (backwards compatibility)"""
    return get_gpu_memory_config()

def main():
    parser = argparse.ArgumentParser(description='Train BEVNeXt-SAM2 Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--force-config', action='store_true', help='Force use of config file instead of auto-detection')
    args = parser.parse_args()
    
    # Load config with dynamic GPU detection
    if args.config and args.force_config:
        print("🔧 Using forced configuration from file")
        with open(args.config, 'r') as f:
            config = json.load(f)
    elif args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print("📁 Using configuration from file")
        except:
            print("⚠️  Config file error, falling back to auto-detection")
            config = get_gpu_memory_config()
    else:
        config = get_gpu_memory_config()
    
    # Apply memory optimizations based on configuration
    apply_memory_optimizations(config)
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"✓ CUDA available - {torch.cuda.device_count()} GPU(s) detected")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
        print(f"  Using GPU: {torch.cuda.current_device()}")
        
        # Print memory usage
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  Memory allocated: {allocated:.1f} MB")
        print(f"  Memory reserved: {reserved:.1f} MB")
    else:
        print("⚠ CUDA not available - using CPU")
    
    # Print configuration summary
    print(f"\n📊 Configuration Summary:")
    print(f"  Model size: {config['d_model']}d, {config['num_transformer_layers']} layers")
    print(f"  BEV resolution: {config['bev_size'][0]}×{config['bev_size'][1]}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient checkpointing: {config.get('gradient_checkpointing', False)}")
    print(f"  Memory fraction: {config.get('memory_fraction', 0.8)}")
    
    # Create trainer
    trainer = Trainer(config, use_mixed_precision=args.mixed_precision)
    
    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed training from epoch {trainer.epoch}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    import math  # Need this for positional encoding
    main()