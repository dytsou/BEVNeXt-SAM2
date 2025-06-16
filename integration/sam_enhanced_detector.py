"""
SAM-Enhanced BEV Detector

This module enhances BEVNeXt's detection capabilities by incorporating
SAM2's segmentation features into the detection pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

import sys
sys.path.append('..')

from sam2_module.modeling.sam2_base import SAM2Base
from bevnext.mmdet3d.models import build_detector


class SAMEnhancedBEVDetector(nn.Module):
    """
    Enhanced BEV detector that incorporates SAM2 features for improved detection.
    
    This module uses SAM2's image encoder features to enhance the BEV features
    before detection, potentially improving detection accuracy.
    """
    
    def __init__(
        self,
        bev_config: Dict,
        sam2_checkpoint: str,
        sam2_model_cfg: str,
        fusion_dim: int = 256,
        device: str = 'cuda'
    ):
        """
        Initialize SAM-Enhanced BEV Detector.
        
        Args:
            bev_config: Configuration dict for BEVNeXt model
            sam2_checkpoint: Path to SAM2 checkpoint
            sam2_model_cfg: Path to SAM2 config
            fusion_dim: Dimension for feature fusion
            device: Device to run on
        """
        super().__init__()
        self.device = device
        
        # Build BEVNeXt detector
        self.bev_detector = build_detector(bev_config)
        
        # Load SAM2 image encoder
        from sam2_module.build_sam import build_sam2
        sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint)
        self.sam2_encoder = sam2_model.image_encoder
        self.sam2_encoder.eval()  # Use SAM2 encoder in eval mode
        
        # Feature fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                self.sam2_encoder.neck.d_model + bev_config.get('bev_channels', 256),
                fusion_dim,
                kernel_size=1
            ),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Adapter to match BEV feature dimensions
        self.bev_adapter = nn.Conv2d(
            fusion_dim,
            bev_config.get('bev_channels', 256),
            kernel_size=1
        )
        
    def extract_sam_features(
        self,
        images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from SAM2 encoder.
        
        Args:
            images: Input images [B, N, C, H, W] where N is number of views
            
        Returns:
            Dictionary of SAM2 features
        """
        B, N, C, H, W = images.shape
        
        # Reshape for batch processing
        images_flat = images.view(B * N, C, H, W)
        
        # Extract SAM2 features
        with torch.no_grad():
            sam_features = self.sam2_encoder(images_flat)
            
        # Get the high-resolution features
        if isinstance(sam_features, dict):
            feat = sam_features['image_embeddings']
        else:
            feat = sam_features
            
        # Reshape back to multi-view
        _, C_feat, H_feat, W_feat = feat.shape
        feat = feat.view(B, N, C_feat, H_feat, W_feat)
        
        return {
            'sam_features': feat,
            'feature_shape': (H_feat, W_feat)
        }
    
    def fuse_features(
        self,
        bev_features: torch.Tensor,
        sam_features: torch.Tensor,
        view_transform_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse SAM2 features with BEV features.
        
        Args:
            bev_features: BEV features [B, C, H, W]
            sam_features: SAM2 features [B, N, C, H, W]
            view_transform_matrix: Optional transformation matrix
            
        Returns:
            Enhanced BEV features
        """
        B, N, C_sam, H_sam, W_sam = sam_features.shape
        B, C_bev, H_bev, W_bev = bev_features.shape
        
        # Project SAM features to BEV space
        # This is a simplified version - actual implementation would use
        # proper view transformation
        sam_bev = sam_features.mean(dim=1)  # Simple average across views
        
        # Resize to match BEV resolution
        if (H_sam, W_sam) != (H_bev, W_bev):
            sam_bev = F.interpolate(
                sam_bev,
                size=(H_bev, W_bev),
                mode='bilinear',
                align_corners=False
            )
        
        # Concatenate and fuse
        combined = torch.cat([bev_features, sam_bev], dim=1)
        fused = self.fusion_conv(combined)
        
        # Adapt back to BEV feature space
        enhanced_bev = self.bev_adapter(fused)
        
        # Residual connection
        enhanced_bev = enhanced_bev + bev_features
        
        return enhanced_bev
    
    def forward(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of SAM-enhanced BEV detector.
        
        Args:
            img: Multi-view images [B, N, C, H, W]
            img_metas: Image metadata
            **kwargs: Additional arguments for BEV detector
            
        Returns:
            Detection results with enhanced features
        """
        # Extract SAM2 features
        sam_feat_dict = self.extract_sam_features(img)
        sam_features = sam_feat_dict['sam_features']
        
        # Get intermediate BEV features from detector
        # This requires modifying the BEV detector to return intermediate features
        # For now, we'll do a standard forward pass
        
        # Run BEV detector
        if hasattr(self.bev_detector, 'extract_feat'):
            # Extract features
            feat_dict = self.bev_detector.extract_feat(img, img_metas)
            
            # Enhance BEV features with SAM features
            if 'bev_feat' in feat_dict:
                feat_dict['bev_feat'] = self.fuse_features(
                    feat_dict['bev_feat'],
                    sam_features
                )
            
            # Complete detection with enhanced features
            if hasattr(self.bev_detector, 'pts_bbox_head'):
                results = self.bev_detector.pts_bbox_head(feat_dict['bev_feat'])
            else:
                results = feat_dict
        else:
            # Fallback to standard forward
            results = self.bev_detector(img, img_metas, **kwargs)
        
        # Add SAM features to results for potential downstream use
        results['sam_features'] = sam_features
        
        return results
    
    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        **kwargs
    ) -> List[Dict]:
        """
        Test-time forward pass.
        
        Args:
            img: Input images
            img_metas: Image metadata
            
        Returns:
            List of detection results
        """
        # Get predictions
        outputs = self.forward(img, img_metas, **kwargs)
        
        # Format results
        results = []
        batch_size = len(img_metas)
        
        for i in range(batch_size):
            result = {}
            
            # Extract detections for this sample
            if 'boxes_3d' in outputs:
                result['boxes_3d'] = outputs['boxes_3d'][i]
                result['scores_3d'] = outputs['scores'][i]
                result['labels_3d'] = outputs['labels'][i]
            
            # Include SAM features if needed
            if 'sam_features' in outputs:
                result['sam_features'] = outputs['sam_features'][i]
            
            results.append(result)
        
        return results 