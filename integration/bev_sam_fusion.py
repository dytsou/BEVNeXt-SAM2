"""
BEV-SAM Fusion Module

This module combines BEVNeXt's Bird's Eye View 3D object detection with 
SAM2's segmentation capabilities for enhanced perception.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append('..')

from sam2_module.build_sam import build_sam2
from sam2_module.sam2_image_predictor import SAM2ImagePredictor


class BEVSAMFusion(nn.Module):
    """
    Fusion module that combines BEVNeXt detections with SAM2 segmentation.
    
    This module takes 3D object detections from BEVNeXt and uses SAM2 to
    provide detailed instance segmentation masks for each detected object.
    """
    
    def __init__(
        self, 
        sam2_checkpoint: str,
        sam2_model_cfg: str,
        device: str = 'cuda'
    ):
        """
        Initialize BEV-SAM Fusion module.
        
        Args:
            sam2_checkpoint: Path to SAM2 model checkpoint
            sam2_model_cfg: Path to SAM2 model configuration
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        
        # Initialize SAM2
        self.sam2_predictor = SAM2ImagePredictor(
            build_sam2(sam2_model_cfg, sam2_checkpoint).to(device)
        )
        
    def project_3d_to_2d(
        self, 
        boxes_3d: torch.Tensor, 
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor
    ) -> torch.Tensor:
        """
        Project 3D bounding boxes to 2D image plane.
        
        Args:
            boxes_3d: 3D bounding boxes from BEVNeXt [N, 7] (x, y, z, w, h, l, yaw)
            camera_intrinsics: Camera intrinsic matrix [3, 3]
            camera_extrinsics: Camera extrinsic matrix [4, 4]
            
        Returns:
            boxes_2d: 2D bounding boxes [N, 4] (x1, y1, x2, y2)
        """
        # Convert 3D box corners to 2D
        corners_3d = self._get_3d_box_corners(boxes_3d)  # [N, 8, 3]
        
        # Transform to camera coordinates
        corners_cam = self._world_to_cam(corners_3d, camera_extrinsics)
        
        # Project to image plane
        corners_2d = self._cam_to_image(corners_cam, camera_intrinsics)
        
        # Get 2D bounding boxes from projected corners
        x_min = corners_2d[..., 0].min(dim=-1)[0]
        y_min = corners_2d[..., 1].min(dim=-1)[0]
        x_max = corners_2d[..., 0].max(dim=-1)[0]
        y_max = corners_2d[..., 1].max(dim=-1)[0]
        
        boxes_2d = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        
        return boxes_2d
    
    def forward(
        self,
        image: torch.Tensor,
        bev_detections: Dict[str, torch.Tensor],
        camera_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining BEV detections with SAM segmentation.
        
        Args:
            image: Input image tensor [B, C, H, W]
            bev_detections: Dictionary containing:
                - 'boxes_3d': 3D bounding boxes [N, 7]
                - 'scores': Confidence scores [N]
                - 'labels': Class labels [N]
            camera_params: Camera parameters for projection
                
        Returns:
            Dictionary containing:
                - 'boxes_3d': Original 3D boxes
                - 'boxes_2d': Projected 2D boxes
                - 'masks': Segmentation masks from SAM2
                - 'scores': Detection scores
                - 'labels': Class labels
        """
        # Project 3D boxes to 2D
        boxes_2d = self.project_3d_to_2d(
            bev_detections['boxes_3d'],
            camera_params['intrinsics'],
            camera_params['extrinsics']
        )
        
        # Use SAM2 to get segmentation masks
        with torch.inference_mode():
            self.sam2_predictor.set_image(image[0].permute(1, 2, 0).cpu().numpy())
            
            masks = []
            for box in boxes_2d:
                # Use box prompt for SAM2
                mask, score, _ = self.sam2_predictor.predict(
                    box=box.cpu().numpy(),
                    multimask_output=False
                )
                masks.append(torch.from_numpy(mask[0]).to(self.device))
            
            masks = torch.stack(masks) if masks else torch.empty(0, *image.shape[-2:])
        
        return {
            'boxes_3d': bev_detections['boxes_3d'],
            'boxes_2d': boxes_2d,
            'masks': masks,
            'scores': bev_detections['scores'],
            'labels': bev_detections['labels']
        }
    
    def _get_3d_box_corners(self, boxes_3d: torch.Tensor) -> torch.Tensor:
        """Convert 3D box parameters to corner points."""
        # Implementation for converting box parameters to 8 corner points
        # This is a simplified version - implement based on BEVNeXt's format
        pass
    
    def _world_to_cam(
        self, 
        points_world: torch.Tensor,
        extrinsics: torch.Tensor
    ) -> torch.Tensor:
        """Transform points from world to camera coordinates."""
        # Add homogeneous coordinate
        ones = torch.ones_like(points_world[..., :1])
        points_homo = torch.cat([points_world, ones], dim=-1)
        
        # Apply extrinsic transformation
        points_cam = torch.matmul(points_homo, extrinsics.T)
        
        return points_cam[..., :3]
    
    def _cam_to_image(
        self,
        points_cam: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """Project camera points to image plane."""
        # Divide by depth
        points_2d_homo = points_cam / points_cam[..., 2:3]
        
        # Apply intrinsic matrix
        points_2d = torch.matmul(points_2d_homo, intrinsics.T)
        
        return points_2d[..., :2] 