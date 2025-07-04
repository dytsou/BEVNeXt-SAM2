"""
Fusion Model Visualizer

Comprehensive visualization tools for BEVNeXt-SAM2 fusion model results.
Creates visualizations for 3D detection, 2D segmentation, and their combination.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont


class FusionVisualizer:
    """
    Comprehensive visualizer for fusion model results.
    
    Creates visualizations for:
    - 3D bounding boxes in BEV and perspective views
    - 2D segmentation masks overlaid on images
    - Multi-view synchronized visualizations
    - Training progress and metrics
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.class_colors = {
            'car': [255, 0, 0],        # Red
            'truck': [0, 255, 0],      # Green  
            'bus': [0, 0, 255],        # Blue
            'trailer': [255, 255, 0],  # Yellow
            'motorcycle': [255, 0, 255], # Magenta
            'bicycle': [0, 255, 255],  # Cyan
            'pedestrian': [255, 128, 0] # Orange
        }
        
        self.class_names = list(self.class_colors.keys())
        
        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def visualize_sample(
        self,
        images: torch.Tensor,
        predictions: Dict,
        targets: Dict,
        img_meta: Dict,
        save_path: Union[str, Path],
        show_confidence: bool = True,
        confidence_threshold: float = 0.3
    ):
        """
        Create comprehensive visualization for a single sample.
        
        Args:
            images: Multi-view images [N, C, H, W]
            predictions: Model predictions
            targets: Ground truth targets
            img_meta: Image metadata
            save_path: Path to save visualization
            show_confidence: Whether to show confidence scores
            confidence_threshold: Minimum confidence to display
        """
        
        # Create figure with subplots
        n_views = images.shape[0]
        fig = plt.figure(figsize=(20, 4 * n_views + 8))
        
        # Main visualization grid
        gs = fig.add_gridspec(n_views + 2, 4, height_ratios=[1] * n_views + [1.5, 1])
        
        # Visualize each camera view
        for view_idx in range(n_views):
            self.visualize_camera_view(
                fig, gs, view_idx,
                images[view_idx], predictions, targets, img_meta[view_idx],
                show_confidence, confidence_threshold
            )
        
        # Bird's Eye View visualization
        ax_bev = fig.add_subplot(gs[n_views, :2])
        self.visualize_bev(ax_bev, predictions, targets, img_meta)
        
        # 3D visualization
        ax_3d = fig.add_subplot(gs[n_views, 2:], projection='3d')
        self.visualize_3d(ax_3d, predictions, targets)
        
        # Metrics and information
        ax_info = fig.add_subplot(gs[n_views + 1, :])
        self.visualize_info(ax_info, predictions, targets, img_meta)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def visualize_camera_view(
        self,
        fig: plt.Figure,
        gs,
        view_idx: int,
        image: torch.Tensor,
        predictions: Dict,
        targets: Dict,
        img_meta: Dict,
        show_confidence: bool,
        confidence_threshold: float
    ):
        """Visualize single camera view with detections and segmentation."""
        
        # Convert image to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
            
        # Create subplot
        ax = fig.add_subplot(gs[view_idx, :])
        ax.imshow(image_np)
        ax.set_title(f"Camera View: {img_meta.get('cam_name', f'view_{view_idx}')}")
        ax.axis('off')
        
        # Overlay segmentation masks if available
        if 'masks_2d' in predictions and predictions['masks_2d'] is not None:
            self.overlay_segmentation_masks(ax, predictions['masks_2d'], view_idx)
        
        # Project and draw 3D boxes
        if 'boxes_3d' in predictions and len(predictions['boxes_3d']) > 0:
            self.draw_projected_boxes(
                ax, predictions['boxes_3d'], predictions.get('scores_3d'),
                predictions.get('labels_3d'), img_meta,
                show_confidence, confidence_threshold, is_prediction=True
            )
        
        # Draw ground truth boxes
        if 'gt_bboxes_3d' in targets and len(targets['gt_bboxes_3d']) > 0:
            self.draw_projected_boxes(
                ax, targets['gt_bboxes_3d'], None,
                targets.get('gt_labels_3d'), img_meta,
                False, 0.0, is_prediction=False
            )
    
    def overlay_segmentation_masks(
        self,
        ax: plt.Axes,
        masks: torch.Tensor,
        view_idx: int,
        alpha: float = 0.5
    ):
        """Overlay segmentation masks on image."""
        
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = masks
            
        # Handle different mask formats
        if masks_np.ndim == 4:  # [N_views, H, W] or [N_instances, N_views, H, W]
            if view_idx < masks_np.shape[0]:
                mask = masks_np[view_idx]
            else:
                return
        elif masks_np.ndim == 3:  # [H, W] or [N_instances, H, W]
            mask = masks_np[0] if masks_np.shape[0] > 1 else masks_np
        else:
            mask = masks_np
            
        # Create colored overlay
        if mask.max() > 0:
            # Normalize mask
            mask_norm = (mask > 0.5).astype(np.float32)
            
            # Create colored mask
            h, w = mask_norm.shape[-2:]
            colored_mask = np.zeros((h, w, 4))
            colored_mask[:, :, :3] = [1.0, 0.0, 0.0]  # Red color
            colored_mask[:, :, 3] = mask_norm * alpha  # Alpha channel
            
            ax.imshow(colored_mask, alpha=alpha)
    
    def draw_projected_boxes(
        self,
        ax: plt.Axes,
        boxes_3d: torch.Tensor,
        scores: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        img_meta: Dict,
        show_confidence: bool,
        confidence_threshold: float,
        is_prediction: bool = True
    ):
        """Draw projected 3D boxes on image."""
        
        if isinstance(boxes_3d, torch.Tensor):
            boxes_3d = boxes_3d.cpu().numpy()
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Project 3D boxes to 2D
        boxes_2d = self.project_3d_to_2d(boxes_3d, img_meta)
        
        for i, box_2d in enumerate(boxes_2d):
            # Check confidence threshold
            if scores is not None and scores[i] < confidence_threshold:
                continue
                
            # Get class color
            class_idx = labels[i] if labels is not None else 0
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown'
            color = np.array(self.class_colors.get(class_name, [128, 128, 128])) / 255.0
            
            # Draw box
            line_style = '-' if is_prediction else '--'
            line_width = 2 if is_prediction else 3
            
            # Draw 2D projection (simplified as rectangle)
            x_min, y_min, x_max, y_max = box_2d
            rect = Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=line_width, edgecolor=color, facecolor='none',
                linestyle=line_style
            )
            ax.add_patch(rect)
            
            # Add label and confidence
            label_text = class_name
            if show_confidence and scores is not None:
                label_text += f': {scores[i]:.2f}'
            if not is_prediction:
                label_text = f'GT: {label_text}'
                
            ax.text(
                x_min, y_min - 5, label_text,
                fontsize=8, color=color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )
    
    def project_3d_to_2d(
        self,
        boxes_3d: np.ndarray,
        img_meta: Dict
    ) -> np.ndarray:
        """Project 3D boxes to 2D image coordinates."""
        
        # Simplified projection - in practice, use proper camera calibration
        # box format: [x, y, z, w, l, h, yaw]
        
        boxes_2d = []
        
        # Get camera parameters
        cam_intrinsic = img_meta.get('cam_intrinsic', np.eye(3))
        
        for box_3d in boxes_3d:
            x, y, z, w, l, h, yaw = box_3d
            
            # Simple projection assuming front camera
            # In practice, use proper transformation matrices
            img_x = 400 + x * 20  # Scale and offset
            img_y = 300 - y * 20
            img_w = w * 30
            img_h = h * 30
            
            x_min = max(0, img_x - img_w/2)
            y_min = max(0, img_y - img_h/2)  
            x_max = min(800, img_x + img_w/2)
            y_max = min(448, img_y + img_h/2)
            
            boxes_2d.append([x_min, y_min, x_max, y_max])
        
        return np.array(boxes_2d)
    
    def visualize_bev(
        self,
        ax: plt.Axes,
        predictions: Dict,
        targets: Dict,
        img_metas: List[Dict]
    ):
        """Visualize Bird's Eye View with 3D boxes."""
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Bird\'s Eye View')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Draw ego vehicle
        ego_rect = Rectangle((-2, -1), 4, 2, facecolor='black', alpha=0.8)
        ax.add_patch(ego_rect)
        ax.text(0, 0, 'EGO', ha='center', va='center', color='white', fontweight='bold')
        
        # Draw prediction boxes
        if 'boxes_3d' in predictions and len(predictions['boxes_3d']) > 0:
            self.draw_bev_boxes(
                ax, predictions['boxes_3d'], 
                predictions.get('scores_3d'), predictions.get('labels_3d'),
                is_prediction=True
            )
        
        # Draw ground truth boxes
        if 'gt_bboxes_3d' in targets and len(targets['gt_bboxes_3d']) > 0:
            self.draw_bev_boxes(
                ax, targets['gt_bboxes_3d'],
                None, targets.get('gt_labels_3d'),
                is_prediction=False
            )
    
    def draw_bev_boxes(
        self,
        ax: plt.Axes,
        boxes_3d: torch.Tensor,
        scores: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        is_prediction: bool = True
    ):
        """Draw 3D boxes in Bird's Eye View."""
        
        if isinstance(boxes_3d, torch.Tensor):
            boxes_3d = boxes_3d.cpu().numpy()
        if scores is not None and isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        for i, box in enumerate(boxes_3d):
            x, y, z, w, l, h, yaw = box
            
            # Get class color
            class_idx = labels[i] if labels is not None else 0
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown'
            color = np.array(self.class_colors.get(class_name, [128, 128, 128])) / 255.0
            
            # Create rotated rectangle
            corners = self.get_box_corners_2d(x, y, w, l, yaw)
            
            # Draw box
            line_style = '-' if is_prediction else '--'
            line_width = 2 if is_prediction else 3
            alpha = 0.7 if is_prediction else 0.5
            
            polygon = Polygon(corners, closed=True, fill=False, 
                            edgecolor=color, linewidth=line_width,
                            linestyle=line_style, alpha=alpha)
            ax.add_patch(polygon)
            
            # Add direction arrow
            arrow_length = max(w, l) / 2
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)
            ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, 
                    fc=color, ec=color, alpha=alpha)
            
            # Add label
            label_text = class_name
            if scores is not None:
                label_text += f': {scores[i]:.2f}'
            if not is_prediction:
                label_text = f'GT: {label_text}'
                
            ax.text(x, y + l/2 + 1, label_text, ha='center', va='bottom',
                   fontsize=8, color=color, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    def get_box_corners_2d(
        self,
        x: float, y: float, w: float, l: float, yaw: float
    ) -> np.ndarray:
        """Get 2D corners of rotated box."""
        
        # Box corners in local coordinate system
        corners_local = np.array([
            [-w/2, -l/2],
            [w/2, -l/2],
            [w/2, l/2],
            [-w/2, l/2]
        ])
        
        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # Rotate and translate corners
        corners_global = corners_local @ rotation_matrix.T
        corners_global[:, 0] += x
        corners_global[:, 1] += y
        
        return corners_global
    
    def visualize_3d(
        self,
        ax: plt.Axes,
        predictions: Dict,
        targets: Dict
    ):
        """Visualize 3D boxes in 3D space."""
        
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-2, 8)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Visualization')
        
        # Draw prediction boxes
        if 'boxes_3d' in predictions and len(predictions['boxes_3d']) > 0:
            self.draw_3d_boxes(
                ax, predictions['boxes_3d'],
                predictions.get('labels_3d'),
                is_prediction=True
            )
        
        # Draw ground truth boxes
        if 'gt_bboxes_3d' in targets and len(targets['gt_bboxes_3d']) > 0:
            self.draw_3d_boxes(
                ax, targets['gt_bboxes_3d'],
                targets.get('gt_labels_3d'),
                is_prediction=False
            )
    
    def draw_3d_boxes(
        self,
        ax: plt.Axes,
        boxes_3d: torch.Tensor,
        labels: Optional[torch.Tensor],
        is_prediction: bool = True
    ):
        """Draw 3D boxes in 3D plot."""
        
        if isinstance(boxes_3d, torch.Tensor):
            boxes_3d = boxes_3d.cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        for i, box in enumerate(boxes_3d):
            x, y, z, w, l, h, yaw = box
            
            # Get class color
            class_idx = labels[i] if labels is not None else 0
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else 'unknown'
            color = np.array(self.class_colors.get(class_name, [128, 128, 128])) / 255.0
            
            # Get 3D box corners
            corners_3d = self.get_box_corners_3d(x, y, z, w, l, h, yaw)
            
            # Draw box edges
            line_style = '-' if is_prediction else '--'
            alpha = 0.8 if is_prediction else 0.6
            
            # Bottom face
            ax.plot3D(*corners_3d[[0, 1, 2, 3, 0]].T, color=color, linestyle=line_style, alpha=alpha)
            # Top face  
            ax.plot3D(*corners_3d[[4, 5, 6, 7, 4]].T, color=color, linestyle=line_style, alpha=alpha)
            # Vertical edges
            for j in range(4):
                ax.plot3D(*corners_3d[[j, j+4]].T, color=color, linestyle=line_style, alpha=alpha)
    
    def get_box_corners_3d(
        self,
        x: float, y: float, z: float,
        w: float, l: float, h: float, yaw: float
    ) -> np.ndarray:
        """Get 3D corners of rotated box."""
        
        # Box corners in local coordinate system
        corners_local = np.array([
            [-w/2, -l/2, -h/2],  # Bottom face
            [w/2, -l/2, -h/2],
            [w/2, l/2, -h/2],
            [-w/2, l/2, -h/2],
            [-w/2, -l/2, h/2],   # Top face
            [w/2, -l/2, h/2],
            [w/2, l/2, h/2],
            [-w/2, l/2, h/2]
        ])
        
        # Rotation matrix around Z-axis
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Rotate and translate corners
        corners_global = corners_local @ rotation_matrix.T
        corners_global[:, 0] += x
        corners_global[:, 1] += y
        corners_global[:, 2] += z
        
        return corners_global
    
    def visualize_info(
        self,
        ax: plt.Axes,
        predictions: Dict,
        targets: Dict,
        img_metas: List[Dict]
    ):
        """Visualize information and statistics."""
        
        ax.axis('off')
        
        # Gather statistics
        n_pred_boxes = len(predictions.get('boxes_3d', []))
        n_gt_boxes = len(targets.get('gt_bboxes_3d', []))
        n_pred_masks = 1 if predictions.get('masks_2d') is not None else 0
        n_gt_masks = 1 if targets.get('gt_masks') is not None else 0
        
        # Create info text
        info_text = f"""
        Sample Information:
        • Predicted 3D Boxes: {n_pred_boxes}
        • Ground Truth 3D Boxes: {n_gt_boxes}
        • Predicted Masks: {n_pred_masks}
        • Ground Truth Masks: {n_gt_masks}
        • Camera Views: {len(img_metas)}
        
        Legend:
        • Solid lines: Predictions
        • Dashed lines: Ground Truth
        • Colors: Different object classes
        """
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add class legend
        legend_elements = []
        for class_name, color in self.class_colors.items():
            color_norm = np.array(color) / 255.0
            legend_elements.append(
                plt.Line2D([0], [0], color=color_norm, lw=2, label=class_name)
            )
        
        ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1, 0.5))
    
    def visualize_training_progress(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Union[str, Path]
    ):
        """Visualize training progress and metrics."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss curves
        if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
            axes[0, 0].plot(metrics_history['train_loss'], label='Train')
            axes[0, 0].plot(metrics_history['val_loss'], label='Validation')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # mAP curves
        if 'val_mAP_3d' in metrics_history:
            axes[0, 1].plot(metrics_history['val_mAP_3d'])
            axes[0, 1].set_title('3D Detection mAP')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Segmentation metrics
        if 'val_IoU' in metrics_history:
            axes[0, 2].plot(metrics_history['val_IoU'], label='IoU')
            if 'val_F1' in metrics_history:
                axes[0, 2].plot(metrics_history['val_F1'], label='F1')
            axes[0, 2].set_title('Segmentation Metrics')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Score')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rate' in metrics_history:
            axes[1, 0].plot(metrics_history['learning_rate'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Consistency metrics
        if 'val_consistency' in metrics_history:
            axes[1, 1].plot(metrics_history['val_consistency'])
            axes[1, 1].set_title('Cross-Modal Consistency')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Consistency Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Overall score
        if 'val_overall_score' in metrics_history:
            axes[1, 2].plot(metrics_history['val_overall_score'])
            axes[1, 2].set_title('Overall Performance Score')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_visualization(
        self,
        results_dict: Dict[str, Dict],
        save_path: Union[str, Path]
    ):
        """Create comparison visualization between different models/methods."""
        
        methods = list(results_dict.keys())
        n_methods = len(methods)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Metric comparison
        metrics = ['3d_mAP', '2d_IoU', 'consistency_score', 'overall_score']
        method_scores = {method: [] for method in methods}
        
        for metric in metrics:
            for method in methods:
                score = results_dict[method].get(metric, 0.0)
                method_scores[method].append(score)
        
        # Bar plot comparison
        x = np.arange(len(metrics))
        width = 0.8 / n_methods
        
        for i, method in enumerate(methods):
            axes[0, 0].bar(x + i * width, method_scores[method], 
                          width, label=method, alpha=0.8)
        
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Comparison')
        axes[0, 0].set_xticks(x + width * (n_methods - 1) / 2)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax_radar = plt.subplot(2, 2, 2, projection='polar')
        
        for method in methods:
            values = method_scores[method] + [method_scores[method][0]]  # Complete the circle
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=method)
            ax_radar.fill(angles, values, alpha=0.25)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Performance Radar Chart')
        ax_radar.legend()
        
        # Processing time comparison (if available)
        if all('processing_time' in results_dict[method] for method in methods):
            times = [results_dict[method]['processing_time'] for method in methods]
            axes[1, 0].bar(methods, times)
            axes[1, 0].set_title('Processing Time Comparison')
            axes[1, 0].set_ylabel('Time (ms)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison (if available)
        if all('memory_usage' in results_dict[method] for method in methods):
            memory = [results_dict[method]['memory_usage'] for method in methods]
            axes[1, 1].bar(methods, memory)
            axes[1, 1].set_title('Memory Usage Comparison')
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_video_visualization(
        self,
        sequence_results: List[Dict],
        save_path: Union[str, Path],
        fps: int = 10
    ):
        """Create video visualization from sequence of results."""
        
        import cv2
        
        # Create temporary frame images
        temp_dir = self.output_dir / 'temp_frames'
        temp_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        
        for i, frame_result in enumerate(sequence_results):
            frame_path = temp_dir / f'frame_{i:06d}.png'
            
            # Create frame visualization
            self.visualize_sample(
                frame_result['images'],
                frame_result['predictions'],
                frame_result['targets'],
                frame_result['img_meta'],
                frame_path
            )
            
            frame_paths.append(str(frame_path))
        
        # Create video from frames
        if frame_paths:
            # Read first frame to get dimensions
            first_frame = cv2.imread(frame_paths[0])
            height, width, _ = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
            
            # Write frames
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                video_writer.write(frame)
            
            video_writer.release()
        
        # Clean up temporary frames
        for frame_path in frame_paths:
            os.remove(frame_path)
        temp_dir.rmdir()
        
    def export_to_html(
        self,
        results: Dict,
        save_path: Union[str, Path]
    ):
        """Export visualization results to interactive HTML."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BEVNeXt-SAM2 Fusion Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
                .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
                .image-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin-top: 20px; }}
                .image-card {{ text-align: center; }}
                .image-card img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>BEVNeXt-SAM2 Fusion Model Results</h1>
                
                <div class="metrics">
                    <div class="metric-card">
                        <h3>3D Detection</h3>
                        <p>mAP: {results.get('3d_mAP', 0.0):.3f}</p>
                        <p>NDS: {results.get('3d_NDS', 0.0):.3f}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>2D Segmentation</h3>
                        <p>IoU: {results.get('2d_IoU', 0.0):.3f}</p>
                        <p>F1: {results.get('2d_F1', 0.0):.3f}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Consistency</h3>
                        <p>Spatial Alignment: {results.get('consistency_spatial', 0.0):.3f}</p>
                        <p>Detection-Segmentation: {results.get('consistency_det_seg', 0.0):.3f}</p>
                    </div>
                    
                    <div class="metric-card">
                        <h3>Overall</h3>
                        <p>Score: {results.get('overall_score', 0.0):.3f}</p>
                    </div>
                </div>
                
                <div class="image-gallery">
                    <!-- Visualization images would be embedded here -->
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)