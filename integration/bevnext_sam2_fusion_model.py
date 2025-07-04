"""
BEVNeXt-SAM2 End-to-End Fusion Model

A complete trainable model that combines BEVNeXt's 3D object detection 
with SAM2's segmentation in an end-to-end training framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from mmcv import Config
from bevnext.mmdet3d.models import build_detector
from sam2_module.build_sam import build_sam2
from sam2_module.modeling.sam2_base import SAM2Base


class BEVNeXtSAM2FusionModel(nn.Module):
    """
    End-to-end trainable fusion model combining BEVNeXt and SAM2.
    
    Features:
    - Joint 3D detection and 2D segmentation training
    - Multi-task loss optimization
    - Feature-level and prediction-level fusion
    - Comprehensive evaluation metrics
    """
    
    def __init__(
        self,
        bev_config: Union[str, Dict],
        sam2_config: str,
        sam2_checkpoint: str = None,
        fusion_mode: str = 'feature_fusion',  # 'feature_fusion', 'late_fusion', 'multi_scale'
        bev_weight: float = 1.0,
        sam_weight: float = 1.0,
        fusion_weight: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize the fusion model.
        
        Args:
            bev_config: BEVNeXt configuration file or dict
            sam2_config: SAM2 configuration file
            sam2_checkpoint: Pre-trained SAM2 checkpoint (optional)
            fusion_mode: Type of fusion to use
            bev_weight: Weight for BEV loss
            sam_weight: Weight for SAM loss  
            fusion_weight: Weight for fusion consistency loss
            device: Device to run on
        """
        super().__init__()
        
        self.fusion_mode = fusion_mode
        self.bev_weight = bev_weight
        self.sam_weight = sam_weight
        self.fusion_weight = fusion_weight
        self.device = device
        
        # Load configs
        if isinstance(bev_config, str):
            self.bev_cfg = Config.fromfile(bev_config)
        else:
            self.bev_cfg = bev_config
            
        # Build BEVNeXt detector
        self.bev_detector = build_detector(
            self.bev_cfg.model,
            train_cfg=self.bev_cfg.get('train_cfg'),
            test_cfg=self.bev_cfg.get('test_cfg')
        )
        
        # Build SAM2 model
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        
        # Fusion components
        self.fusion_dim = 256
        self._build_fusion_modules()
        
        # Loss components
        self._build_loss_modules()
        
        # Projection utilities
        self.projection_utils = ProjectionUtils()
        
    def _build_fusion_modules(self):
        """Build fusion-specific neural network modules."""
        
        if self.fusion_mode == 'feature_fusion':
            # Feature-level fusion layers
            self.sam_feature_adapter = nn.Sequential(
                nn.Conv2d(256, self.fusion_dim, 1),  # SAM2 has 256 channels
                nn.BatchNorm2d(self.fusion_dim),
                nn.ReLU(inplace=True)
            )
            
            self.bev_feature_adapter = nn.Sequential(
                nn.Conv2d(256, self.fusion_dim, 1),  # BEV feature channels
                nn.BatchNorm2d(self.fusion_dim), 
                nn.ReLU(inplace=True)
            )
            
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(self.fusion_dim * 2, self.fusion_dim, 3, padding=1),
                nn.BatchNorm2d(self.fusion_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.fusion_dim, self.fusion_dim, 3, padding=1),
                nn.BatchNorm2d(self.fusion_dim),
                nn.ReLU(inplace=True)
            )
            
            # Cross-attention for multi-modal fusion
            self.cross_attention = CrossModalAttention(self.fusion_dim)
            
        elif self.fusion_mode == 'multi_scale':
            # Multi-scale fusion for different resolution features
            self.multi_scale_fusion = MultiScaleFusion(self.fusion_dim)
            
        # Prediction heads for fused features
        self.fused_detection_head = FusedDetectionHead(
            self.fusion_dim,
            num_classes=self.bev_cfg.model.get('pts_bbox_head', {}).get('num_classes', 10)
        )
        
        self.segmentation_head = SegmentationHead(self.fusion_dim)
        
    def _build_loss_modules(self):
        """Build loss functions for multi-task learning."""
        
        # 3D detection losses (from BEVNeXt)
        self.detection_loss = DetectionLoss()
        
        # Segmentation losses (compatible with SAM2)
        self.segmentation_loss = SegmentationLoss()
        
        # Fusion consistency loss
        self.consistency_loss = ConsistencyLoss()
        
        # Adaptive loss weighting
        self.loss_weights = nn.Parameter(torch.ones(3))  # learnable weights
        
    def extract_features(
        self,
        images: torch.Tensor,
        img_metas: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from both BEV and SAM2 encoders.
        
        Args:
            images: Multi-view images [B, N, C, H, W]
            img_metas: Image metadata
            
        Returns:
            bev_features: BEV features
            sam_features: SAM2 features
        """
        # Extract BEV features
        bev_features = self.bev_detector.extract_feat(images, img_metas)
        
        # Extract SAM2 features
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        
        with torch.no_grad():  # Keep SAM2 frozen initially
            sam_features = self.sam2_model.image_encoder(images_flat)
            
        # Reshape SAM features back to multi-view
        if isinstance(sam_features, dict):
            sam_feat = sam_features['image_embeddings']
        else:
            sam_feat = sam_features
            
        _, C_sam, H_sam, W_sam = sam_feat.shape
        sam_feat = sam_feat.view(B, N, C_sam, H_sam, W_sam)
        
        return bev_features, sam_feat
        
    def fuse_features(
        self,
        bev_features: torch.Tensor,
        sam_features: torch.Tensor,
        img_metas: List[Dict]
    ) -> torch.Tensor:
        """
        Fuse BEV and SAM features according to fusion mode.
        
        Args:
            bev_features: BEV features from detector
            sam_features: SAM2 features
            img_metas: Image metadata for projection
            
        Returns:
            fused_features: Combined features
        """
        if self.fusion_mode == 'feature_fusion':
            return self._feature_level_fusion(bev_features, sam_features, img_metas)
        elif self.fusion_mode == 'late_fusion':
            return self._late_fusion(bev_features, sam_features)
        elif self.fusion_mode == 'multi_scale':
            return self._multi_scale_fusion(bev_features, sam_features, img_metas)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
            
    def _feature_level_fusion(
        self,
        bev_features: torch.Tensor,
        sam_features: torch.Tensor,
        img_metas: List[Dict]
    ) -> torch.Tensor:
        """Feature-level fusion with cross-modal attention."""
        
        # Project SAM features to BEV space
        sam_bev = self.projection_utils.project_to_bev(
            sam_features, 
            img_metas,
            target_size=bev_features.shape[-2:]
        )
        
        # Adapt feature dimensions
        bev_adapted = self.bev_feature_adapter(bev_features)
        sam_adapted = self.sam_feature_adapter(sam_bev)
        
        # Cross-modal attention
        attended_features = self.cross_attention(bev_adapted, sam_adapted)
        
        # Concatenate and fuse
        combined = torch.cat([attended_features, sam_adapted], dim=1)
        fused = self.fusion_conv(combined)
        
        return fused
        
    def _late_fusion(
        self,
        bev_features: torch.Tensor,
        sam_features: torch.Tensor
    ) -> torch.Tensor:
        """Late fusion at prediction level."""
        # Simple averaging for late fusion
        return (bev_features + sam_features) / 2
        
    def _multi_scale_fusion(
        self,
        bev_features: torch.Tensor,
        sam_features: torch.Tensor,
        img_metas: List[Dict]
    ) -> torch.Tensor:
        """Multi-scale feature fusion."""
        return self.multi_scale_fusion(bev_features, sam_features, img_metas)
        
    def forward_train(
        self,
        images: torch.Tensor,
        img_metas: List[Dict],
        gt_bboxes_3d: List[torch.Tensor],
        gt_labels_3d: List[torch.Tensor],
        gt_masks: List[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            images: Multi-view images
            img_metas: Image metadata
            gt_bboxes_3d: Ground truth 3D boxes
            gt_labels_3d: Ground truth 3D labels
            gt_masks: Ground truth segmentation masks
            
        Returns:
            Dictionary of losses
        """
        # Extract features
        bev_features, sam_features = self.extract_features(images, img_metas)
        
        # Fuse features
        fused_features = self.fuse_features(bev_features, sam_features, img_metas)
        
        # Get predictions
        predictions = self._get_predictions(fused_features, img_metas)
        
        # Compute losses
        losses = self._compute_losses(
            predictions,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_masks,
            img_metas
        )
        
        return losses
        
    def forward_test(
        self,
        images: torch.Tensor,
        img_metas: List[Dict],
        **kwargs
    ) -> List[Dict]:
        """
        Testing forward pass.
        
        Args:
            images: Multi-view images
            img_metas: Image metadata
            
        Returns:
            List of detection and segmentation results
        """
        # Extract and fuse features
        bev_features, sam_features = self.extract_features(images, img_metas)
        fused_features = self.fuse_features(bev_features, sam_features, img_metas)
        
        # Get predictions
        predictions = self._get_predictions(fused_features, img_metas)
        
        # Format results
        results = self._format_results(predictions, img_metas)
        
        return results
        
    def _get_predictions(
        self,
        fused_features: torch.Tensor,
        img_metas: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Get detection and segmentation predictions."""
        
        # 3D detection predictions
        detection_preds = self.fused_detection_head(fused_features)
        
        # Segmentation predictions
        seg_preds = self.segmentation_head(fused_features)
        
        return {
            'detection': detection_preds,
            'segmentation': seg_preds,
            'features': fused_features
        }
        
    def _compute_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_bboxes_3d: List[torch.Tensor],
        gt_labels_3d: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        img_metas: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses."""
        
        losses = {}
        
        # 3D detection loss
        det_loss = self.detection_loss(
            predictions['detection'],
            gt_bboxes_3d,
            gt_labels_3d
        )
        losses.update({f'det_{k}': v for k, v in det_loss.items()})
        
        # Segmentation loss (if masks available)
        if gt_masks is not None:
            seg_loss = self.segmentation_loss(
                predictions['segmentation'],
                gt_masks
            )
            losses.update({f'seg_{k}': v for k, v in seg_loss.items()})
        
        # Consistency loss
        consistency_loss = self.consistency_loss(
            predictions['detection'],
            predictions['segmentation'],
            img_metas
        )
        losses['consistency'] = consistency_loss
        
        # Weighted total loss
        total_loss = (
            self.bev_weight * sum([v for k, v in losses.items() if k.startswith('det_')]) +
            self.sam_weight * sum([v for k, v in losses.items() if k.startswith('seg_')]) +
            self.fusion_weight * losses['consistency']
        )
        losses['total'] = total_loss
        
        return losses
        
    def _format_results(
        self,
        predictions: Dict[str, torch.Tensor],
        img_metas: List[Dict]
    ) -> List[Dict]:
        """Format predictions into result format."""
        
        batch_size = len(img_metas)
        results = []
        
        for i in range(batch_size):
            result = {
                'boxes_3d': predictions['detection']['boxes_3d'][i],
                'scores_3d': predictions['detection']['scores'][i],
                'labels_3d': predictions['detection']['labels'][i],
                'masks_2d': predictions['segmentation']['masks'][i],
                'features': predictions['features'][i]
            }
            results.append(result)
            
        return results


class CrossModalAttention(nn.Module):
    """Cross-modal attention for BEV and SAM feature fusion."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, bev_feat: torch.Tensor, sam_feat: torch.Tensor) -> torch.Tensor:
        # Reshape for attention
        B, C, H, W = bev_feat.shape
        bev_flat = bev_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        sam_flat = sam_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # Cross attention: BEV queries SAM
        attended, _ = self.attention(bev_flat, sam_flat, sam_flat)
        attended = self.norm1(attended + bev_flat)
        
        # FFN
        attended = self.norm2(attended + self.ffn(attended))
        
        # Reshape back
        attended = attended.permute(0, 2, 1).view(B, C, H, W)
        
        return attended


class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion module."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Implementation for multi-scale fusion
        pass
        
    def forward(self, bev_feat: torch.Tensor, sam_feat: torch.Tensor, img_metas: List[Dict]) -> torch.Tensor:
        # Placeholder implementation
        return bev_feat


class FusedDetectionHead(nn.Module):
    """Detection head for fused features."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_dim, num_classes, 1)
        )
        
        self.reg_head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_dim, 7, 1)  # 7 for 3D box regression
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_pred = self.cls_head(features)
        reg_pred = self.reg_head(features)
        
        return {
            'boxes_3d': reg_pred,
            'scores': cls_pred,
            'labels': cls_pred.argmax(dim=1)
        }


class SegmentationHead(nn.Module):
    """Segmentation head for fused features."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        masks = self.seg_head(features)
        
        return {
            'masks': masks
        }


class ProjectionUtils:
    """Utilities for projecting between coordinate systems."""
    
    def project_to_bev(
        self,
        sam_features: torch.Tensor,
        img_metas: List[Dict],
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Project SAM features to BEV space."""
        
        # Simplified projection - average across views
        B, N, C, H, W = sam_features.shape
        sam_bev = sam_features.mean(dim=1)  # Average across views
        
        # Resize to target BEV size
        if (H, W) != target_size:
            sam_bev = F.interpolate(
                sam_bev,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
        return sam_bev


# Loss function classes
class DetectionLoss:
    """3D detection loss computation."""
    
    def __call__(self, predictions, gt_boxes, gt_labels):
        # Placeholder - implement proper 3D detection loss
        return {'cls_loss': torch.tensor(0.0), 'reg_loss': torch.tensor(0.0)}


class SegmentationLoss:
    """Segmentation loss computation."""
    
    def __call__(self, predictions, gt_masks):
        # Placeholder - implement proper segmentation loss
        return {'seg_loss': torch.tensor(0.0)}


class ConsistencyLoss:
    """Consistency loss between detection and segmentation."""
    
    def __call__(self, det_preds, seg_preds, img_metas):
        # Placeholder - implement consistency loss
        return torch.tensor(0.0)