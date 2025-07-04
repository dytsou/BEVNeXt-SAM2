"""
Fusion Model Evaluator

Comprehensive evaluation for BEVNeXt-SAM2 fusion model that computes
metrics for both 3D object detection and 2D segmentation tasks.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve


class FusionEvaluator:
    """
    Comprehensive evaluator for fusion model.
    
    Computes metrics for:
    - 3D object detection (mAP, NDS, etc.)
    - 2D segmentation (IoU, F1, etc.)
    - Cross-modal consistency
    """
    
    def __init__(self, eval_config: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            eval_config: Optional evaluation configuration file
        """
        self.eval_config = eval_config
        
        # 3D detection thresholds
        self.iou_thresholds_3d = [0.5, 0.7]
        self.distance_thresholds = [5, 10, 20, 30, 40, 50]  # meters
        
        # 2D segmentation thresholds
        self.iou_thresholds_2d = np.arange(0.5, 1.0, 0.05)
        
        # Class names (should match dataset)
        self.class_names = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']
        
    def evaluate(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of predictions vs targets.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of ground truth dictionaries
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # 3D detection evaluation
        detection_metrics = self.evaluate_3d_detection(predictions, targets)
        results.update({f"3d_{k}": v for k, v in detection_metrics.items()})
        
        # 2D segmentation evaluation
        segmentation_metrics = self.evaluate_2d_segmentation(predictions, targets)
        results.update({f"2d_{k}": v for k, v in segmentation_metrics.items()})
        
        # Cross-modal consistency evaluation
        consistency_metrics = self.evaluate_consistency(predictions, targets)
        results.update({f"consistency_{k}": v for k, v in consistency_metrics.items()})
        
        # Overall metrics
        results['overall_score'] = self.compute_overall_score(results)
        
        return results
        
    def evaluate_3d_detection(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate 3D object detection performance."""
        
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_gt_boxes = []
        all_gt_labels = []
        
        # Collect all predictions and targets
        for pred, target in zip(predictions, targets):
            if 'boxes_3d' in pred and len(pred['boxes_3d']) > 0:
                all_pred_boxes.append(pred['boxes_3d'].cpu().numpy())
                all_pred_scores.append(pred['scores_3d'].cpu().numpy())
                all_pred_labels.append(pred['labels_3d'].cpu().numpy())
            else:
                all_pred_boxes.append(np.zeros((0, 7)))
                all_pred_scores.append(np.zeros(0))
                all_pred_labels.append(np.zeros(0, dtype=int))
                
            if 'gt_bboxes_3d' in target and len(target['gt_bboxes_3d']) > 0:
                all_gt_boxes.append(target['gt_bboxes_3d'].cpu().numpy())
                all_gt_labels.append(target['gt_labels_3d'].cpu().numpy())
            else:
                all_gt_boxes.append(np.zeros((0, 7)))
                all_gt_labels.append(np.zeros(0, dtype=int))
        
        # Compute mAP for different IoU thresholds
        metrics = {}
        for iou_thresh in self.iou_thresholds_3d:
            ap_per_class = []
            
            for class_idx, class_name in enumerate(self.class_names):
                ap = self.compute_ap_3d(
                    all_pred_boxes, all_pred_scores, all_pred_labels,
                    all_gt_boxes, all_gt_labels,
                    class_idx, iou_thresh
                )
                ap_per_class.append(ap)
                metrics[f'AP_{class_name}@{iou_thresh}'] = ap
            
            metrics[f'mAP@{iou_thresh}'] = np.mean(ap_per_class)
        
        # Overall mAP
        metrics['mAP_3d'] = np.mean([metrics[f'mAP@{thresh}'] for thresh in self.iou_thresholds_3d])
        
        # Distance-based evaluation
        for dist_thresh in self.distance_thresholds:
            recall = self.compute_recall_at_distance(
                all_pred_boxes, all_gt_boxes, dist_thresh
            )
            metrics[f'Recall@{dist_thresh}m'] = recall
        
        # NDS (nuScenes Detection Score) if applicable
        if len(self.distance_thresholds) > 0:
            nds = self.compute_nds(all_pred_boxes, all_pred_scores, all_gt_boxes)
            metrics['NDS'] = nds
        
        return metrics
        
    def evaluate_2d_segmentation(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate 2D segmentation performance."""
        
        metrics = {}
        
        all_pred_masks = []
        all_gt_masks = []
        
        # Collect all masks
        for pred, target in zip(predictions, targets):
            if 'masks_2d' in pred and pred['masks_2d'] is not None:
                pred_masks = pred['masks_2d']
                if isinstance(pred_masks, torch.Tensor):
                    pred_masks = pred_masks.cpu().numpy()
                all_pred_masks.append(pred_masks)
            else:
                all_pred_masks.append(None)
                
            if 'gt_masks' in target and target['gt_masks'] is not None:
                gt_masks = target['gt_masks']
                if isinstance(gt_masks, list):
                    # Multi-view masks
                    gt_masks = torch.stack(gt_masks) if len(gt_masks) > 0 else torch.zeros(1, 1, 1)
                if isinstance(gt_masks, torch.Tensor):
                    gt_masks = gt_masks.cpu().numpy()
                all_gt_masks.append(gt_masks)
            else:
                all_gt_masks.append(None)
        
        # Filter out None masks
        valid_pairs = [(p, g) for p, g in zip(all_pred_masks, all_gt_masks) 
                      if p is not None and g is not None]
        
        if len(valid_pairs) == 0:
            # No segmentation data available
            return {
                'IoU': 0.0,
                'F1': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'mAP_seg': 0.0
            }
        
        # Compute IoU metrics
        ious = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for pred_mask, gt_mask in valid_pairs:
            # Ensure same shape
            if pred_mask.shape != gt_mask.shape:
                continue
                
            # Binarize predictions
            pred_binary = (pred_mask > 0.5).astype(np.uint8)
            gt_binary = (gt_mask > 0.5).astype(np.uint8)
            
            # Compute metrics for each view/instance
            for i in range(min(pred_binary.shape[0], gt_binary.shape[0])):
                pred_view = pred_binary[i] if pred_binary.ndim > 2 else pred_binary
                gt_view = gt_binary[i] if gt_binary.ndim > 2 else gt_binary
                
                iou = self.compute_iou_2d(pred_view, gt_view)
                precision, recall, f1 = self.compute_seg_metrics(pred_view, gt_view)
                
                ious.append(iou)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
        
        if len(ious) > 0:
            metrics['IoU'] = np.mean(ious)
            metrics['Precision'] = np.mean(precisions)
            metrics['Recall'] = np.mean(recalls)
            metrics['F1'] = np.mean(f1_scores)
            
            # mAP for segmentation at different IoU thresholds
            ap_segs = []
            for thresh in self.iou_thresholds_2d:
                ap_seg = np.mean([1.0 if iou >= thresh else 0.0 for iou in ious])
                ap_segs.append(ap_seg)
            metrics['mAP_seg'] = np.mean(ap_segs)
        else:
            metrics.update({
                'IoU': 0.0,
                'F1': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'mAP_seg': 0.0
            })
        
        return metrics
        
    def evaluate_consistency(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate consistency between 3D detection and 2D segmentation."""
        
        consistency_scores = []
        spatial_alignment_scores = []
        
        for pred, target in zip(predictions, targets):
            # Check if both 3D and 2D predictions exist
            if ('boxes_3d' not in pred or 'masks_2d' not in pred or 
                pred['boxes_3d'] is None or pred['masks_2d'] is None):
                continue
                
            boxes_3d = pred['boxes_3d']
            masks_2d = pred['masks_2d']
            
            if len(boxes_3d) == 0 or masks_2d is None:
                continue
            
            # Compute spatial consistency
            # This would involve projecting 3D boxes to 2D and comparing with masks
            spatial_score = self.compute_spatial_consistency(boxes_3d, masks_2d, target.get('img_metas'))
            spatial_alignment_scores.append(spatial_score)
            
            # Compute detection-segmentation consistency
            det_seg_score = self.compute_detection_segmentation_consistency(boxes_3d, masks_2d)
            consistency_scores.append(det_seg_score)
        
        metrics = {}
        if len(consistency_scores) > 0:
            metrics['det_seg_consistency'] = np.mean(consistency_scores)
        else:
            metrics['det_seg_consistency'] = 0.0
            
        if len(spatial_alignment_scores) > 0:
            metrics['spatial_alignment'] = np.mean(spatial_alignment_scores)
        else:
            metrics['spatial_alignment'] = 0.0
        
        return metrics
        
    def compute_ap_3d(
        self,
        pred_boxes: List[np.ndarray],
        pred_scores: List[np.ndarray], 
        pred_labels: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        gt_labels: List[np.ndarray],
        class_idx: int,
        iou_threshold: float
    ) -> float:
        """Compute Average Precision for 3D detection."""
        
        # Collect predictions and ground truths for specific class
        all_pred_scores = []
        all_pred_matched = []
        total_gt = 0
        
        for i in range(len(pred_boxes)):
            # Filter predictions for this class
            class_mask = pred_labels[i] == class_idx
            class_pred_boxes = pred_boxes[i][class_mask]
            class_pred_scores = pred_scores[i][class_mask]
            
            # Filter ground truths for this class  
            gt_class_mask = gt_labels[i] == class_idx
            class_gt_boxes = gt_boxes[i][gt_class_mask]
            total_gt += len(class_gt_boxes)
            
            if len(class_pred_boxes) == 0:
                continue
                
            # Compute IoU matrix
            if len(class_gt_boxes) > 0:
                ious = self.compute_iou_matrix_3d(class_pred_boxes, class_gt_boxes)
                
                # Match predictions to ground truths
                matched = np.zeros(len(class_pred_boxes), dtype=bool)
                gt_matched = np.zeros(len(class_gt_boxes), dtype=bool)
                
                # Sort predictions by score
                sorted_indices = np.argsort(class_pred_scores)[::-1]
                
                for pred_idx in sorted_indices:
                    if len(class_gt_boxes) == 0:
                        break
                        
                    # Find best matching GT
                    gt_ious = ious[pred_idx]
                    valid_gts = np.where(~gt_matched)[0]
                    
                    if len(valid_gts) == 0:
                        break
                        
                    best_gt_idx = valid_gts[np.argmax(gt_ious[valid_gts])]
                    best_iou = gt_ious[best_gt_idx]
                    
                    if best_iou >= iou_threshold:
                        matched[pred_idx] = True
                        gt_matched[best_gt_idx] = True
                        
                all_pred_matched.extend(matched.tolist())
            else:
                all_pred_matched.extend([False] * len(class_pred_boxes))
                
            all_pred_scores.extend(class_pred_scores.tolist())
        
        if len(all_pred_scores) == 0 or total_gt == 0:
            return 0.0
            
        # Sort by score
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        all_pred_matched = np.array(all_pred_matched)[sorted_indices]
        
        # Compute precision and recall
        tp = np.cumsum(all_pred_matched)
        fp = np.cumsum(~all_pred_matched)
        
        precision = tp / (tp + fp)
        recall = tp / total_gt
        
        # Compute AP using trapezoidal rule
        ap = np.trapz(precision, recall) if len(recall) > 1 else 0.0
        
        return ap
        
    def compute_iou_matrix_3d(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of 3D boxes."""
        
        ious = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                ious[i, j] = self.compute_iou_3d(box1, box2)
                
        return ious
        
    def compute_iou_3d(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute 3D IoU between two boxes."""
        # Simplified 3D IoU computation
        # box format: [x, y, z, w, l, h, yaw]
        
        # For simplicity, use 2D IoU in BEV space
        return self.compute_iou_bev(box1, box2)
        
    def compute_iou_bev(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU in Bird's Eye View."""
        # Extract x, y, w, l, yaw
        x1, y1, w1, l1, yaw1 = box1[0], box1[1], box1[3], box1[4], box1[6]
        x2, y2, w2, l2, yaw2 = box2[0], box2[1], box2[3], box2[4], box2[6]
        
        # For simplicity, assume no rotation and use axis-aligned boxes
        # In practice, you'd need proper rotated box IoU
        
        # Convert to corner coordinates
        corners1 = np.array([
            [x1 - w1/2, y1 - l1/2],
            [x1 + w1/2, y1 - l1/2],
            [x1 + w1/2, y1 + l1/2],
            [x1 - w1/2, y1 + l1/2]
        ])
        
        corners2 = np.array([
            [x2 - w2/2, y2 - l2/2],
            [x2 + w2/2, y2 - l2/2],
            [x2 + w2/2, y2 + l2/2],
            [x2 - w2/2, y2 + l2/2]
        ])
        
        # Compute intersection area (simplified)
        x_left = max(corners1[:, 0].min(), corners2[:, 0].min())
        y_top = max(corners1[:, 1].min(), corners2[:, 1].min())
        x_right = min(corners1[:, 0].max(), corners2[:, 0].max())
        y_bottom = min(corners1[:, 1].max(), corners2[:, 1].max())
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * l1
        area2 = w2 * l2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def compute_iou_2d(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute 2D IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
            
        return intersection / union
        
    def compute_seg_metrics(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score for segmentation."""
        
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        tp = np.logical_and(pred_flat == 1, gt_flat == 1).sum()
        fp = np.logical_and(pred_flat == 1, gt_flat == 0).sum()
        fn = np.logical_and(pred_flat == 0, gt_flat == 1).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
        
    def compute_recall_at_distance(
        self,
        pred_boxes: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        distance_threshold: float
    ) -> float:
        """Compute recall at specific distance threshold."""
        
        total_gt = 0
        detected_gt = 0
        
        for pred, gt in zip(pred_boxes, gt_boxes):
            if len(gt) == 0:
                continue
                
            total_gt += len(gt)
            
            if len(pred) == 0:
                continue
                
            # Compute distances from origin
            gt_distances = np.sqrt(gt[:, 0]**2 + gt[:, 1]**2)
            valid_gt = gt_distances <= distance_threshold
            
            if not valid_gt.any():
                continue
                
            # Count detections within distance
            pred_distances = np.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
            valid_pred = pred_distances <= distance_threshold
            
            # Simple matching based on distance (simplified)
            detected_gt += min(valid_gt.sum(), valid_pred.sum())
        
        return detected_gt / total_gt if total_gt > 0 else 0.0
        
    def compute_nds(
        self,
        pred_boxes: List[np.ndarray],
        pred_scores: List[np.ndarray],
        gt_boxes: List[np.ndarray]
    ) -> float:
        """Compute nuScenes Detection Score (simplified version)."""
        
        # This is a simplified NDS computation
        # Real NDS involves multiple metrics and weighted averaging
        
        map_score = 0.0
        tp_metrics = []
        
        # Compute mean AP across classes and thresholds
        for class_idx in range(len(self.class_names)):
            class_aps = []
            for thresh in self.iou_thresholds_3d:
                ap = self.compute_ap_3d(
                    pred_boxes, pred_scores, 
                    [np.full(len(pb), class_idx) for pb in pred_boxes],
                    gt_boxes,
                    [np.full(len(gb), class_idx) for gb in gt_boxes],
                    class_idx, thresh
                )
                class_aps.append(ap)
            map_score += np.mean(class_aps)
        
        map_score /= len(self.class_names)
        
        # Add other TP metrics (simplified)
        tp_metrics = [0.8, 0.7, 0.9, 0.6, 0.8]  # Placeholder values
        
        # Weighted combination
        nds = (map_score + np.mean(tp_metrics)) / 2
        
        return nds
        
    def compute_spatial_consistency(
        self,
        boxes_3d: torch.Tensor,
        masks_2d: torch.Tensor,
        img_metas: Optional[List[Dict]] = None
    ) -> float:
        """Compute spatial consistency between 3D boxes and 2D masks."""
        
        # This would involve projecting 3D boxes to 2D and comparing with masks
        # For now, return a placeholder score
        return 0.7
        
    def compute_detection_segmentation_consistency(
        self,
        boxes_3d: torch.Tensor,
        masks_2d: torch.Tensor
    ) -> float:
        """Compute consistency between detection and segmentation predictions."""
        
        # Simple consistency metric based on overlap
        # In practice, this would be more sophisticated
        
        if len(boxes_3d) == 0 or masks_2d is None:
            return 0.0
            
        # Count non-empty masks vs number of detections
        if isinstance(masks_2d, torch.Tensor):
            mask_count = (masks_2d.sum(dim=(-2, -1)) > 0).sum().item()
        else:
            mask_count = len(masks_2d)
            
        det_count = len(boxes_3d)
        
        # Simple consistency score
        if det_count == 0 and mask_count == 0:
            return 1.0
        elif det_count == 0 or mask_count == 0:
            return 0.0
        else:
            return min(det_count, mask_count) / max(det_count, mask_count)
        
    def compute_overall_score(self, metrics: Dict[str, float]) -> float:
        """Compute overall performance score."""
        
        # Weighted combination of key metrics
        weights = {
            '3d_mAP_3d': 0.4,
            '2d_F1': 0.3,
            'consistency_spatial_alignment': 0.2,
            'consistency_det_seg_consistency': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
        
    def save_evaluation_report(
        self,
        metrics: Dict[str, float],
        save_path: str
    ):
        """Save detailed evaluation report."""
        
        # Create evaluation plots
        self.plot_evaluation_results(metrics, save_path)
        
        # Save metrics to file
        import json
        with open(f"{save_path}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def plot_evaluation_results(
        self,
        metrics: Dict[str, float],
        save_path: str
    ):
        """Create evaluation result plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 3D detection metrics
        detection_metrics = {k.replace('3d_', ''): v for k, v in metrics.items() if k.startswith('3d_')}
        if detection_metrics:
            axes[0, 0].bar(detection_metrics.keys(), detection_metrics.values())
            axes[0, 0].set_title('3D Detection Metrics')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2D segmentation metrics
        seg_metrics = {k.replace('2d_', ''): v for k, v in metrics.items() if k.startswith('2d_')}
        if seg_metrics:
            axes[0, 1].bar(seg_metrics.keys(), seg_metrics.values())
            axes[0, 1].set_title('2D Segmentation Metrics')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Consistency metrics
        consistency_metrics = {k.replace('consistency_', ''): v for k, v in metrics.items() if k.startswith('consistency_')}
        if consistency_metrics:
            axes[1, 0].bar(consistency_metrics.keys(), consistency_metrics.values())
            axes[1, 0].set_title('Consistency Metrics')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall score
        axes[1, 1].bar(['Overall Score'], [metrics.get('overall_score', 0.0)])
        axes[1, 1].set_title('Overall Performance')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()