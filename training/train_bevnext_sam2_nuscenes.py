#!/usr/bin/env python3
"""
Enhanced BEVNeXt-SAM2 Training Script with nuScenes v1.0 Integration

This script provides comprehensive training pipeline specifically adapted for nuScenes v1.0
dataset characteristics including token-based associations, multi-modal sensor data,
and 23 object categories.

Author: Senior Python Programmer & Autonomous Driving Dataset Expert
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
    print("Warning: TensorBoard not available, logging will be limited")
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import numpy as np
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    print("Warning: Matplotlib not available, plotting will be disabled")
from collections import defaultdict, OrderedDict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import nuScenes components
try:
    from nuscenes_dataset_v2 import (
        NuScenesMultiModalDataset,
        NuScenesConfig,
        create_nuscenes_dataloader,
        NuScenesDataAnalyzer
    )
    NUSCENES_AVAILABLE = True
    print("Enhanced nuScenes dataset module imported successfully")
except ImportError as e:
    NUSCENES_AVAILABLE = False
    print(f"Enhanced nuScenes dataset not available: {e}")

# Import SAM2 components
try:
    from sam2_module.build_sam import build_sam2
    from sam2_module.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    print("SAM2 module imported successfully")
except ImportError as e:
    SAM2_AVAILABLE = False
    print(f"SAM2 not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedBEVNeXtSAM2Model(nn.Module):
    """
    Enhanced BEVNeXt-SAM2 model specifically designed for nuScenes multi-modal data

    Supports:
    - Multi-camera fusion (6 cameras)
    - LiDAR point cloud processing
    - Radar data integration
    - 23 nuScenes object categories
    - Token-based data associations
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.get('gradient_checkpointing', False)
        self.use_lidar = config.get('use_lidar', False)
        self.use_radar = config.get('use_radar', False)

        # nuScenes specific configuration
        self.num_cameras = len(config.get('camera_names', ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']))
        self.num_classes = config.get('num_classes', 23)  # nuScenes has 23 categories

        # Initialize SAM2 for feature extraction
        self._init_sam2()

        # Build model components
        self.camera_backbone = self._build_camera_backbone()
        if self.use_lidar:
            self.lidar_backbone = self._build_lidar_backbone()
        if self.use_radar:
            self.radar_backbone = self._build_radar_backbone()

        self.bev_transformer = self._build_bev_transformer()
        self.sam2_fusion = self._build_sam2_fusion()
        self.detection_head = self._build_detection_head()

        # Setup loss functions
        self.setup_losses()

        # Enable gradient checkpointing for memory efficiency
        if self.use_gradient_checkpointing:
            logger.info("Gradient checkpointing enabled for memory efficiency")
            self._enable_gradient_checkpointing()

    def _init_sam2(self):
        """Initialize SAM2 for feature extraction"""
        if not SAM2_AVAILABLE:
            logger.warning("SAM2 module not available. Using placeholder features.")
            self.sam2_model = None
            self.sam2_predictor = None
            return

        # SAM2 configuration
        sam2_config = self.config.get('sam2_config')
        sam2_checkpoint = self.config.get('sam2_checkpoint')

        # If no config provided, use placeholder
        if sam2_config is None or sam2_checkpoint is None:
            logger.warning("SAM2 config or checkpoint not provided. Using placeholder features.")
            self.sam2_model = None
            self.sam2_predictor = None
            return

        try:
            # Load SAM2 model
            self.sam2_model = build_sam2(
                config_file=sam2_config,
                ckpt_path=sam2_checkpoint,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Create SAM2 image predictor for feature extraction
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

            logger.info(f"SAM2 model initialized with config: {sam2_config}")

        except Exception as e:
            logger.warning(f"Failed to load SAM2 model: {e}")
            logger.warning("Falling back to placeholder SAM2 features")
            self.sam2_model = None
            self.sam2_predictor = None

    def _build_camera_backbone(self) -> nn.Module:
        """Build multi-camera feature extraction backbone"""
        d_model = self.config['d_model']

        # Shared backbone for all cameras
        backbone = nn.Sequential(
            # Input: [B*N_cams, 3, H, W]
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            # ResNet-like blocks
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),

            # Output projection
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, d_model)
        )

        return backbone

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Module:
        """Create a ResNet-like layer"""
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_block(self, in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
        """Create a ResNet-like block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _build_lidar_backbone(self) -> nn.Module:
        """Build LiDAR point cloud processing backbone"""
        d_model = self.config['d_model']

        # Simple PointNet-like architecture
        return nn.Sequential(
            nn.Conv1d(4, 64, 1),  # Input: [x, y, z, intensity]
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )

    def _build_radar_backbone(self) -> nn.Module:
        """Build radar data processing backbone"""
        d_model = self.config['d_model']

        return nn.Sequential(
            nn.Linear(6, 64),  # Input: [x, y, z, vx, vy, rcs]
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model),
            nn.ReLU(inplace=True)
        )

    def _build_bev_transformer(self) -> nn.Module:
        """Build BEV transformer for spatial reasoning"""
        d_model = self.config['d_model']
        nhead = self.config['nhead']
        num_layers = self.config['num_transformer_layers']

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )

        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_sam2_fusion(self) -> nn.Module:
        """Build SAM2 fusion module"""
        d_model = self.config['d_model']

        return nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 3, padding=1),  # BEV + SAM2 features
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 1)  # Final projection
        )

    def _build_detection_head(self) -> nn.Module:
        """Build 3D object detection head for nuScenes categories"""
        d_model = self.config['d_model']

        return nn.ModuleDict({
            'classification': nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, self.num_classes)
            ),
            'regression': nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 10)  # [x,y,z,w,l,h,yaw,vx,vy,vz]
            ),
            'confidence': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1)
            )
        })

    def setup_losses(self):
        """Setup loss functions for nuScenes training"""
        # Classification loss with class weights for imbalanced data
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='mean')

        # Additional losses for nuScenes
        self.focal_loss = self._build_focal_loss()

    def _build_focal_loss(self, alpha: float = 0.25, gamma: float = 2.0) -> nn.Module:
        """Build focal loss for handling class imbalance"""
        class FocalLoss(nn.Module):
            def __init__(self, alpha=alpha, gamma=gamma):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()

        return FocalLoss()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.camera_backbone, 'children'):
            for module in self.camera_backbone.children():
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = True

        if hasattr(self.bev_transformer, 'layers'):
            for layer in self.bev_transformer.layers:
                if hasattr(layer, 'use_checkpoint'):
                    layer.use_checkpoint = True

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the enhanced model

        Args:
            batch: Dictionary containing:
                - camera_images: [B, N_cams, 3, H, W]
                - lidar_points: List of [N_points, 4] tensors (optional)
                - radar_points: List of [N_points, 6] tensors (optional)
                - camera_intrinsics: [B, N_cams, 3, 3]
                - camera_extrinsics: [B, N_cams, 4, 4]

        Returns:
            Dictionary with predictions
        """
        # Store current batch for SAM2 feature extraction
        self._current_batch = batch

        B = batch['camera_images'].shape[0] if 'camera_images' in batch else batch['gt_boxes'].shape[0]
        device = next(self.parameters()).device

        # Process multi-camera features
        if 'camera_images' in batch:
            camera_features = self._process_camera_features(batch['camera_images'])
        else:
            camera_features = torch.zeros(B, self.num_cameras, self.config['d_model']).to(device)

        # Process LiDAR features (if available)
        if self.use_lidar and 'lidar_points' in batch:
            lidar_features = self._process_lidar_features(batch['lidar_points'])
        else:
            lidar_features = torch.zeros(B, self.config['d_model']).to(device)

        # Process radar features (if available)
        if self.use_radar and 'radar_points' in batch:
            radar_features = self._process_radar_features(batch['radar_points'])
        else:
            radar_features = torch.zeros(B, self.config['d_model']).to(device)

        # Project to BEV space
        bev_features = self._project_to_bev(camera_features, lidar_features, radar_features)

        # Generate SAM2 features using real SAM2 model
        sam2_features = self._generate_sam2_features(bev_features)

        # Fuse BEV and SAM2 features
        fused_features = self._fuse_features(bev_features, sam2_features)

        # Apply BEV transformer
        transformed_features = self._apply_bev_transformer(fused_features)

        # Generate object queries and predictions
        predictions = self._generate_predictions(transformed_features)

        # Clean up batch reference
        delattr(self, '_current_batch')

        return predictions

    def _process_camera_features(self, camera_images: torch.Tensor) -> torch.Tensor:
        """Process multi-camera images"""
        B, N_cams, C, H, W = camera_images.shape

        # Reshape for batch processing
        flat_images = camera_images.view(B * N_cams, C, H, W)

        # Extract features
        if self.training and self.use_gradient_checkpointing:
            flat_features = checkpoint(self.camera_backbone, flat_images, use_reentrant=False)
        else:
            flat_features = self.camera_backbone(flat_images)

        # Reshape back to [B, N_cams, d_model]
        camera_features = flat_features.view(B, N_cams, -1)

        return camera_features

    def _process_lidar_features(self, lidar_points: List[torch.Tensor]) -> torch.Tensor:
        """Process LiDAR point clouds"""
        B = len(lidar_points)
        device = next(self.parameters()).device

        lidar_features = []
        for points in lidar_points:
            if points.numel() == 0:
                # Empty point cloud
                lidar_features.append(torch.zeros(self.config['d_model']).to(device))
            else:
                points = points.to(device)
                # Transpose for conv1d: [4, N_points]
                points_t = points.T.unsqueeze(0)  # [1, 4, N_points]

                if self.training and self.use_gradient_checkpointing:
                    feat = checkpoint(self.lidar_backbone, points_t, use_reentrant=False)
                else:
                    feat = self.lidar_backbone(points_t)

                lidar_features.append(feat.squeeze(0))

        return torch.stack(lidar_features)

    def _process_radar_features(self, radar_points: List[torch.Tensor]) -> torch.Tensor:
        """Process radar point clouds"""
        B = len(radar_points)
        device = next(self.parameters()).device

        radar_features = []
        for points in radar_points:
            if points.numel() == 0:
                radar_features.append(torch.zeros(self.config['d_model']).to(device))
            else:
                points = points.to(device)
                # Average pooling over points
                mean_points = points.mean(dim=0)  # [6]

                if self.training and self.use_gradient_checkpointing:
                    feat = checkpoint(self.radar_backbone, mean_points.unsqueeze(0), use_reentrant=False)
                else:
                    feat = self.radar_backbone(mean_points.unsqueeze(0))

                radar_features.append(feat.squeeze(0))

        return torch.stack(radar_features)

    def _project_to_bev(self, camera_features: torch.Tensor,
                       lidar_features: torch.Tensor,
                       radar_features: torch.Tensor) -> torch.Tensor:
        """Project multi-modal features to BEV space"""
        B = camera_features.shape[0]
        bev_h, bev_w = self.config['bev_size']
        d_model = self.config['d_model']

        # Combine all features
        # Camera features: [B, N_cams, d_model] -> [B, d_model] (average pooling)
        camera_feat = camera_features.mean(dim=1)

        # Combine modalities
        combined_feat = camera_feat + lidar_features + radar_features  # [B, d_model]

        # Project to BEV grid
        bev_features = combined_feat.view(B, d_model, 1, 1).expand(B, d_model, bev_h, bev_w)

        return bev_features

    def _generate_sam2_features(self, bev_features: torch.Tensor) -> torch.Tensor:
        """Generate SAM2 features from camera images"""
        # Get the current batch from the forward pass context
        if not hasattr(self, '_current_batch'):
            # Fallback to placeholder if batch not available
            logger.warning("No camera images available for SAM2 feature extraction, using placeholder")
            return torch.randn_like(bev_features)

        batch = self._current_batch

        # Check if SAM2 is available and camera images exist
        if self.sam2_predictor is None or 'camera_images' not in batch:
            return torch.randn_like(bev_features)

        try:
            return self._extract_sam2_features_from_cameras(batch['camera_images'], bev_features.shape)
        except Exception as e:
            logger.warning(f"SAM2 feature extraction failed: {e}, using placeholder")
            return torch.randn_like(bev_features)

    def _extract_sam2_features_from_cameras(self, camera_images: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Extract SAM2 features from camera images"""
        B, N_cams, C, H, W = camera_images.shape
        target_B, target_C, target_H, target_W = target_shape

        # Process each camera image to extract SAM2 features
        sam2_features_list = []

        for b in range(B):
            batch_features = []

            for cam in range(N_cams):
                # Get single camera image [C, H, W]
                cam_image = camera_images[b, cam]  # [C, H, W]

                # Convert to numpy format for SAM2 (H, W, C) with values [0, 255]
                cam_image_np = cam_image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                cam_image_np = (cam_image_np * 255).astype(np.uint8)

                # Set image in SAM2 predictor and extract features
                with torch.no_grad():
                    self.sam2_predictor.set_image(cam_image_np)
                    # Get image embeddings from SAM2
                    sam2_embed = self.sam2_predictor.get_image_embedding()  # [1, C_sam2, H_sam2, W_sam2]

                    # Resize to target spatial dimensions if needed
                    if sam2_embed.shape[-2:] != (target_H, target_W):
                        sam2_embed = F.interpolate(
                            sam2_embed,
                            size=(target_H, target_W),
                            mode='bilinear',
                            align_corners=False
                        )

                    batch_features.append(sam2_embed.squeeze(0))  # Remove batch dim: [C_sam2, H, W]

            # Average features across cameras
            avg_features = torch.stack(batch_features).mean(dim=0)  # [C_sam2, H, W]

            # Project to target channel dimension
            if avg_features.shape[0] != target_C:
                # Use a simple projection layer
                if not hasattr(self, 'sam2_channel_proj'):
                    self.sam2_channel_proj = nn.Conv2d(
                        avg_features.shape[0],
                        target_C,
                        kernel_size=1
                    ).to(avg_features.device)

                avg_features = self.sam2_channel_proj(avg_features.unsqueeze(0)).squeeze(0)

            sam2_features_list.append(avg_features)

        # Stack batch dimension
        sam2_features = torch.stack(sam2_features_list)  # [B, target_C, target_H, target_W]

        return sam2_features

    def _fuse_features(self, bev_features: torch.Tensor, sam2_features: torch.Tensor) -> torch.Tensor:
        """Fuse BEV and SAM2 features"""
        combined = torch.cat([bev_features, sam2_features], dim=1)
        return self.sam2_fusion(combined)

    def _apply_bev_transformer(self, features: torch.Tensor) -> torch.Tensor:
        """Apply BEV transformer"""
        B, C, H, W = features.shape

        # Flatten spatial dimensions
        flat_features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Add positional encoding
        pos_encoding = self._get_positional_encoding(H, W, C).to(features.device)
        flat_features = flat_features + pos_encoding.unsqueeze(0)

        # Apply transformer
        if self.training and self.use_gradient_checkpointing:
            transformed = checkpoint(self.bev_transformer, flat_features, use_reentrant=False)
        else:
            transformed = self.bev_transformer(flat_features)

        return transformed

    def _get_positional_encoding(self, H: int, W: int, C: int) -> torch.Tensor:
        """Generate 2D positional encoding"""
        pos_h = torch.arange(H).float().unsqueeze(1).repeat(1, W)
        pos_w = torch.arange(W).float().unsqueeze(0).repeat(H, 1)

        pos_h = pos_h / H * 2 - 1
        pos_w = pos_w / W * 2 - 1

        # Create sinusoidal encoding
        div_term = torch.exp(torch.arange(0, C, 2).float() * (-np.log(10000.0) / C))

        pe = torch.zeros(H * W, C)
        pos_flat = torch.stack([pos_h.flatten(), pos_w.flatten()], dim=1)

        pe[:, 0::4] = torch.sin(pos_flat[:, 0:1] * div_term[::2])
        pe[:, 1::4] = torch.cos(pos_flat[:, 0:1] * div_term[::2])
        pe[:, 2::4] = torch.sin(pos_flat[:, 1:2] * div_term[::2])
        pe[:, 3::4] = torch.cos(pos_flat[:, 1:2] * div_term[::2])

        return pe

    def _generate_predictions(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate final predictions"""
        num_queries = self.config['num_queries']

        # Take first num_queries features as object queries
        object_queries = features[:, :num_queries]  # [B, num_queries, C]

        predictions = {
            'classification': self.detection_head['classification'](object_queries),
            'regression': self.detection_head['regression'](object_queries),
            'confidence': self.detection_head['confidence'](object_queries)
        }

        return predictions

    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute training losses with nuScenes-specific adaptations"""
        losses = {}

        # Extract predictions and targets
        pred_cls = predictions['classification']  # [B, num_queries, num_classes]
        pred_reg = predictions['regression']      # [B, num_queries, 10]
        pred_conf = predictions['confidence']     # [B, num_queries, 1]

        target_cls = targets['gt_labels']         # [B, num_queries]
        target_reg = targets['gt_boxes']          # [B, num_queries, 10]
        target_valid = targets['gt_valid']        # [B, num_queries]

        # Classification loss using focal loss for class imbalance
        B, num_queries = target_cls.shape
        pred_cls_flat = pred_cls.reshape(-1, pred_cls.size(-1))
        target_cls_flat = target_cls.reshape(-1)

        # Handle empty annotations - only compute loss for valid targets
        if target_valid.sum() > 0:
            valid_mask = target_valid.flatten()
            pred_cls_valid = pred_cls_flat[valid_mask]
            target_cls_valid = target_cls_flat[valid_mask]
            losses['classification'] = self.focal_loss(pred_cls_valid, target_cls_valid)
        else:
            # No valid targets in this batch, use a small dummy loss
            losses['classification'] = torch.tensor(0.0, device=pred_cls.device)

        # Regression loss (only for valid targets)
        if target_valid.sum() > 0:
            valid_mask = target_valid.flatten()
            pred_reg_valid = pred_reg.reshape(-1, 10)[valid_mask]
            target_reg_valid = target_reg.reshape(-1, 10)[valid_mask]

            losses['regression'] = self.reg_loss(pred_reg_valid, target_reg_valid)
        else:
            losses['regression'] = torch.tensor(0.0, device=pred_reg.device)

        # Confidence loss - ensure target shape matches prediction shape
        B, num_queries = pred_conf.shape[:2]

        # Create a properly shaped target confidence tensor
        if target_valid.numel() == 0 or target_valid.shape[1] == 0:
            # No valid annotations - create all-zero confidence targets
            target_conf = torch.zeros(B, num_queries, 1, device=pred_conf.device)
        else:
            # Ensure target_valid has the correct shape [B, num_queries]
            if target_valid.shape[1] != num_queries:
                # Pad or truncate to match num_queries
                target_conf_temp = torch.zeros(B, num_queries, device=target_valid.device, dtype=target_valid.dtype)
                min_queries = min(target_valid.shape[1], num_queries)
                target_conf_temp[:, :min_queries] = target_valid[:, :min_queries]
                target_conf = target_conf_temp.float().unsqueeze(-1)
            else:
                target_conf = target_valid.float().unsqueeze(-1)  # [B, num_queries, 1]

        losses['confidence'] = self.conf_loss(pred_conf, target_conf)

        # Total loss with weights
        loss_weights = self.config.get('loss_weights', {'cls': 1.0, 'reg': 5.0, 'conf': 1.0})
        losses['total'] = (
            loss_weights['cls'] * losses['classification'] +
            loss_weights['reg'] * losses['regression'] +
            loss_weights['conf'] * losses['confidence']
        )

        return losses


class NuScenesTrainer:
    """Enhanced trainer for nuScenes dataset"""

    def __init__(self, config: Dict, use_mixed_precision: bool = True):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()

        # Initialize data loaders as None first (before any potential failures)
        self.train_loader = None
        self.val_loader = None

        # Initialize training state early
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_stats = defaultdict(list)

        # Setup model
        self.model = EnhancedBEVNeXtSAM2Model(config).to(self.device)

        # Setup optimizer with different learning rates for different components
        try:
            self.optimizer = self._setup_optimizer()
        except Exception as e:
            logger.warning(f"Failed to setup custom optimizer: {e}")
            logger.info("Using default Adam optimizer")
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 1e-4))

        # Setup mixed precision scaler
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")

        # Setup scheduler
        try:
            self.scheduler = self._setup_scheduler()
        except Exception as e:
            logger.warning(f"Failed to setup custom scheduler: {e}")
            logger.info("Using default StepLR scheduler")
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Setup data loaders with error handling
        try:
            self.train_loader = self._create_dataloader('train')
            self.val_loader = self._create_dataloader('val')
            logger.info(f"Data loaders created successfully - Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            logger.warning("Creating dummy data loaders for testing...")
            self._create_dummy_loaders()

        # Setup logging
        self.setup_logging()

    def _create_dummy_loaders(self):
        """Create dummy data loaders for testing when real dataset is not available"""
        from torch.utils.data import TensorDataset, DataLoader

        # Create dummy dataset
        dummy_data = torch.zeros(10, 6, 3, 224, 224)  # [B, N_cams, C, H, W]
        dummy_targets = torch.zeros(10, 100, 10)  # [B, N_objects, 10] (boxes)
        dummy_labels = torch.zeros(10, 100, dtype=torch.long)  # [B, N_objects]
        dummy_valid = torch.zeros(10, 100, dtype=torch.bool)  # [B, N_objects]

        # Create dataset with all components
        class DummyDataset(TensorDataset):
            def __getitem__(self, idx):
                return {
                    'camera_images': dummy_data[idx],
                    'gt_boxes': dummy_targets[idx],
                    'gt_labels': dummy_labels[idx],
                    'gt_valid': dummy_valid[idx],
                    'sample_token': f'dummy_{idx}',
                    'camera_intrinsics': torch.eye(3, 4),
                    'camera_extrinsics': torch.eye(4, 4)
                }

            def __len__(self):
                return len(dummy_data)

        dummy_dataset = DummyDataset(dummy_data)

        # Create data loaders
        self.train_loader = DataLoader(
            dummy_dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            dummy_dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=False,
            num_workers=0
        )

        logger.warning("Using dummy data loaders - nuScenes dataset not available")
        logger.warning("This is for testing only. For real training, ensure nuScenes dataset is properly configured.")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with parameter groups"""
        # Different learning rates for different components
        base_lr = self.config['learning_rate']
        param_groups = []

        # Backbone parameters (lower learning rate)
        backbone_params = []
        backbone_params.extend(self.model.camera_backbone.parameters())
        if hasattr(self.model, 'lidar_backbone'):
            backbone_params.extend(self.model.lidar_backbone.parameters())
        if hasattr(self.model, 'radar_backbone'):
            backbone_params.extend(self.model.radar_backbone.parameters())

        param_groups.append({
            'params': backbone_params,
            'lr': base_lr * 0.1,
            'name': 'backbone'
        })

        # Transformer parameters (medium learning rate)
        param_groups.append({
            'params': self.model.bev_transformer.parameters(),
            'lr': base_lr * 0.5,
            'name': 'transformer'
        })

        # Head parameters (full learning rate)
        head_params = []
        head_params.extend(self.model.sam2_fusion.parameters())
        head_params.extend(self.model.detection_head.parameters())

        param_groups.append({
            'params': head_params,
            'lr': base_lr,
            'name': 'heads'
        })

        return optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay'],
            eps=1e-8
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[group['lr'] for group in self.optimizer.param_groups],
            total_steps=self.config['num_epochs'] * len(self.train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create nuScenes data loader"""
        if not NUSCENES_AVAILABLE:
            raise RuntimeError("nuScenes dataset module not available")

        # Create nuScenes configuration
        nuscenes_config = NuScenesConfig(
            data_root=self.config['data_root'],
            version=self.config.get('nuscenes_version', 'v1.0-trainval'),
            image_size=tuple(self.config['image_size']),
            bev_size=tuple(self.config['bev_size']),
            num_queries=self.config['num_queries'],
            use_lidar=self.config.get('use_lidar', False),
            use_radar=self.config.get('use_radar', False),
            use_camera=self.config.get('use_camera', True),
            use_augmentation=self.config.get('use_augmentation', True) and split == 'train'
        )

        return create_nuscenes_dataloader(
            config=nuscenes_config,
            split=split,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=(split == 'train')
        )

    def setup_logging(self):
        """Setup logging and tensorboard"""
        self.output_dir = Path(self.config['output_dir'])

        try:
            # Create directory with parents and proper permissions
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure the directory is writable by changing permissions
            import subprocess
            import os
            try:
                # Make sure the directory is writable by the current user
                os.chmod(self.output_dir, 0o755)
                # Also ensure parent directories are accessible
                for parent in self.output_dir.parents:
                    if parent.exists():
                        os.chmod(parent, 0o755)
            except (PermissionError, OSError):
                # If we can't change permissions, continue anyway
                pass

            # Setup tensorboard
            if TENSORBOARD_AVAILABLE:
                tb_dir = self.output_dir / 'tensorboard'
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(tb_dir)
            else:
                self.writer = None
                logger.warning("TensorBoard not available, logging will be limited.")

            # Save configuration
            config_path = self.output_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"Logging setup complete. Output directory: {self.output_dir}")

        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to setup logging directory {self.output_dir}: {e}")
            logger.warning("Running without persistent logging")
            self.writer = None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch+1} Train')

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(batch)
                    losses = self.model.compute_loss(predictions, batch)

                # Backward pass with gradient scaling
                self.scaler.scale(losses['total']).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(batch)
                losses = self.model.compute_loss(predictions, batch)

                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update scheduler
            self.scheduler.step()

            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()

            # Calculate metrics
            metrics = self._calculate_metrics(predictions, batch)
            for key, value in metrics.items():
                epoch_metrics[key] += value

            # Update progress bar
            pbar.set_postfix({
                'total': f"{losses['total'].item():.4f}",
                'cls': f"{losses['classification'].item():.4f}",
                'reg': f"{losses['regression'].item():.4f}",
                'conf': f"{losses['confidence'].item():.4f}"
            })

            # Log to tensorboard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            for key, value in losses.items():
                if self.writer:
                    self.writer.add_scalar(f'train_loss/{key}', value.item(), global_step)

            # Clear cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average metrics
        num_batches = len(self.train_loader)
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        avg_metrics = {key: value / num_batches for key, value in epoch_metrics.items()}

        return {**avg_losses, **avg_metrics}

    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(float)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch+1} Val')

            for batch in pbar:
                batch = self._move_batch_to_device(batch)

                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(batch)
                        losses = self.model.compute_loss(predictions, batch)
                else:
                    predictions = self.model(batch)
                    losses = self.model.compute_loss(predictions, batch)

                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value.item()

                # Calculate metrics
                metrics = self._calculate_metrics(predictions, batch)
                for key, value in metrics.items():
                    epoch_metrics[key] += value

                pbar.set_postfix({'val_loss': f"{losses['total'].item():.4f}"})

        # Average metrics
        num_batches = len(self.val_loader)
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        avg_metrics = {key: value / num_batches for key, value in epoch_metrics.items()}

        return {**avg_losses, **avg_metrics}

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                device_batch[key] = [v.to(self.device) for v in value]
            else:
                device_batch[key] = value
        return device_batch

    def _calculate_metrics(self, predictions: Dict, targets: Dict) -> Dict[str, float]:
        """Calculate training metrics"""
        metrics = {}

        # Classification accuracy
        pred_cls = predictions['classification']  # [B, num_queries, num_classes]
        target_cls = targets['gt_labels']         # [B, num_queries]
        target_valid = targets['gt_valid']        # [B, num_queries]

        if target_valid.sum() > 0:
            valid_mask = target_valid.flatten()
            pred_cls_valid = pred_cls.reshape(-1, pred_cls.size(-1))[valid_mask]
            target_cls_valid = target_cls.flatten()[valid_mask]

            pred_labels = pred_cls_valid.argmax(dim=-1)
            accuracy = (pred_labels == target_cls_valid).float().mean()
            metrics['accuracy'] = accuracy.item()
        else:
            metrics['accuracy'] = 0.0

        return metrics

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        # Ensure output directory exists with proper permissions
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            import os
            os.chmod(self.output_dir, 0o755)
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not ensure output directory permissions: {e}")
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_stats': dict(self.training_stats)
        }

        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        try:
            torch.save(checkpoint, latest_path)
        except (PermissionError, OSError) as e:
            logger.warning(f"Failed to save latest checkpoint: {e}")

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            try:
                torch.save(checkpoint, best_path)
                logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
            except (PermissionError, OSError) as e:
                logger.warning(f"Failed to save best checkpoint: {e}")

        # Save epoch checkpoint
        if (self.epoch + 1) % 10 == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{self.epoch+1}.pth'
            try:
                torch.save(checkpoint, epoch_path)
            except (PermissionError, OSError) as e:
                logger.warning(f"Failed to save epoch checkpoint: {e}")

    def train(self):
        """Main training loop"""
        logger.info("Starting enhanced BEVNeXt-SAM2 training with nuScenes dataset...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log epoch results
            logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_metrics['total']:.4f} - "
                f"Val Loss: {val_metrics['total']:.4f} - "
                f"Accuracy: {train_metrics.get('accuracy', 0):.3f} - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Save training stats
            for key, value in train_metrics.items():
                self.training_stats[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.training_stats[f'val_{key}'].append(value)

            # Log to tensorboard
            if self.writer:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'epoch_train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'epoch_val/{key}', value, epoch)

            # Save checkpoint
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']

            self.save_checkpoint(is_best)

        logger.info("Training completed!")
        if self.writer:
            self.writer.close()


def get_enhanced_config(data_root: str = "data/nuscenes") -> Dict:
    """Get enhanced configuration for nuScenes training"""
    return {
        # Data configuration
        'data_root': data_root,
        'nuscenes_version': 'v1.0-trainval',  # Use full dataset instead of mini

        # Model configuration
        'd_model': 128,  # Reduced for memory efficiency
        'nhead': 8,
        'num_transformer_layers': 4,
        'num_classes': 23,  # nuScenes categories
        'num_queries': 100,  # Reduced for memory efficiency

        # SAM2 configuration - Use a simpler approach without external config
        'sam2_config': None,  # Disable external SAM2 config for now
        'sam2_checkpoint': None,  # Disable external checkpoint for now

        # Multi-modal configuration
        'use_camera': True,
        'use_lidar': False,  # Disabled for memory efficiency
        'use_radar': False,  # Disabled for memory efficiency
        'camera_names': ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'],

        # Image and BEV configuration
        'image_size': [224, 224],
        'bev_size': [48, 48],

        # Training configuration
        'batch_size': 1,  # Small batch for memory efficiency
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'num_workers': 2,

        # Loss weights
        'loss_weights': {
            'cls': 1.0,
            'reg': 5.0,
            'conf': 1.0
        },

        # Memory optimization
        'gradient_checkpointing': True,
        'use_augmentation': True,

        # Output
        'output_dir': '/workspace/outputs/training_nuscenes_enhanced'
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Enhanced BEVNeXt-SAM2 Training with nuScenes')
    parser.add_argument('--data-root', default='data/nuscenes', help='Path to nuScenes dataset')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_enhanced_config(args.data_root)

    # Override config with command line arguments
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.data_root != 'data/nuscenes':
        config['data_root'] = args.data_root

    # Validate nuScenes availability
    if not NUSCENES_AVAILABLE:
        logger.error("Enhanced nuScenes dataset module not available!")
        logger.error("Please ensure the nuScenes dataset and dependencies are properly installed.")
        return

    # Create trainer
    try:
        trainer = NuScenesTrainer(config, use_mixed_precision=args.mixed_precision)
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        logger.error("Please check that nuScenes dataset is available and properly configured.")
        return

    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Resumed training from epoch {trainer.epoch}")

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()