#!/usr/bin/env python3
"""
BEVNeXt-SAM2 Model Validation Script

This script validates trained BEVNeXt-SAM2 models on the nuScenes dataset,
providing comprehensive evaluation metrics and visualizations.

Usage:
    python validation/validate_model.py --checkpoint path/to/checkpoint.pth --data-root /path/to/nuscenes
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, OrderedDict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from training.train_bevnext_sam2_nuscenes import EnhancedBEVNeXtSAM2Model
    from training.nuscenes_dataset_v2 import NuScenesMultiModalDataset, NuScenesConfig, create_nuscenes_dataloader
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.detection.config import config_factory
    NUSCENES_AVAILABLE = True
except ImportError as e:
    NUSCENES_AVAILABLE = False
    print(f"Warning: Some modules not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation for BEVNeXt-SAM2"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 data_root: str = "data/nuscenes",
                 config: Dict = None,
                 device: str = None):
        """
        Initialize model validator
        
        Args:
            checkpoint_path: Path to model checkpoint
            data_root: Path to nuScenes dataset
            config: Model configuration
            device: Device to run validation on
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data_root = Path(data_root)
        self.config = config or self._get_default_config()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate inputs
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset not found: {data_root}")
        
        # Setup model and data
        self.model = self._load_model()
        self.val_loader = self._setup_dataloader()
        
        # Initialize metrics
        self.metrics = defaultdict(list)
        self.predictions = []
        self.ground_truth = []
        
        logger.info(f"Model validator initialized:")
        logger.info(f"   └─ Checkpoint: {checkpoint_path}")
        logger.info(f"   └─ Dataset: {data_root}")
        logger.info(f"   └─ Device: {self.device}")
    
    def _get_default_config(self) -> Dict:
        """Get default validation configuration"""
        return {
            'd_model': 128,
            'nhead': 8,
            'num_transformer_layers': 4,
            'num_classes': 23,  # nuScenes categories
            'num_queries': 100,
            'sam2_config': 'sam2_module/configs/sam2/sam2_hiera_s.yaml',
            'use_camera': True,
            'use_lidar': False,
            'use_radar': False,
            'image_size': [224, 224],
            'bev_size': [48, 48],
            'batch_size': 1,
            'num_workers': 2
        }
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint"""
        logger.info("Loading model from checkpoint...")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Create model
            model = EnhancedBEVNeXtSAM2Model(self.config).to(self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model state from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded model state (no epoch info)")
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_dataloader(self) -> torch.utils.data.DataLoader:
        """Setup validation data loader"""
        logger.info("Setting up validation data loader...")
        
        if not NUSCENES_AVAILABLE:
            raise RuntimeError("nuScenes modules not available")
        
        # Create nuScenes configuration
        nuscenes_config = NuScenesConfig(
            data_root=str(self.data_root),
            version='v1.0-trainval',
            image_size=tuple(self.config['image_size']),
            bev_size=tuple(self.config['bev_size']),
            num_queries=self.config['num_queries'],
            use_lidar=self.config.get('use_lidar', False),
            use_radar=self.config.get('use_radar', False),
            use_camera=self.config.get('use_camera', True),
            use_augmentation=False  # No augmentation for validation
        )
        
        return create_nuscenes_dataloader(
            config=nuscenes_config,
            split='val',
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False
        )
    
    def validate(self, max_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Run validation on the model
        
        Args:
            max_samples: Maximum number of samples to validate (None for all)
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info(f"Starting validation on {len(self.val_loader)} batches...")
        
        self.model.eval()
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                if max_samples and total_samples >= max_samples:
                    break
                
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Run inference
                predictions = self.model(batch)
                
                # Compute metrics
                batch_metrics = self._compute_batch_metrics(predictions, batch)
                
                # Store results
                for key, value in batch_metrics.items():
                    self.metrics[key].append(value)
                
                # Store predictions for detailed analysis
                self._store_predictions(predictions, batch, batch_idx)
                
                total_samples += batch['camera_images'].size(0)
        
        # Compute final metrics
        final_metrics = self._compute_final_metrics()
        
        logger.info(f"Validation completed on {total_samples} samples")
        return final_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _compute_batch_metrics(self, predictions: Dict, batch: Dict) -> Dict[str, float]:
        """Compute metrics for a single batch"""
        metrics = {}
        
        # Detection metrics
        if 'pred_boxes' in predictions and 'gt_boxes' in batch:
            # Compute IoU for detection
            pred_boxes = predictions['pred_boxes']  # [B, N, 10]
            gt_boxes = batch['gt_boxes']  # [B, M, 10]
            gt_valid = batch['gt_valid']  # [B, M]
            
            # Compute IoU for each prediction
            batch_ious = []
            for b in range(pred_boxes.size(0)):
                valid_gt = gt_boxes[b][gt_valid[b]]
                if len(valid_gt) > 0:
                    ious = self._compute_3d_iou(pred_boxes[b], valid_gt)
                    batch_ious.extend(ious.flatten().cpu().numpy())
            
            if batch_ious:
                metrics['mean_iou'] = np.mean(batch_ious)
                metrics['max_iou'] = np.max(batch_ious)
        
        # Classification metrics
        if 'pred_logits' in predictions and 'gt_labels' in batch:
            pred_logits = predictions['pred_logits']  # [B, N, num_classes]
            gt_labels = batch['gt_labels']  # [B, M]
            gt_valid = batch['gt_valid']  # [B, M]
            
            # Compute classification accuracy
            pred_labels = pred_logits.argmax(dim=-1)  # [B, N]
            
            batch_acc = []
            for b in range(pred_labels.size(0)):
                valid_gt = gt_labels[b][gt_valid[b]]
                if len(valid_gt) > 0:
                    # Simple accuracy (can be improved with proper matching)
                    acc = (pred_labels[b][:len(valid_gt)] == valid_gt).float().mean().item()
                    batch_acc.append(acc)
            
            if batch_acc:
                metrics['classification_accuracy'] = np.mean(batch_acc)
        
        return metrics
    
    def _compute_3d_iou(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """Compute 3D IoU between predicted and ground truth boxes"""
        # Simplified 3D IoU computation
        # In practice, you'd want to use a proper 3D IoU implementation
        
        # For now, compute center distance as a proxy
        pred_centers = pred_boxes[:, :3]  # [N, 3]
        gt_centers = gt_boxes[:, :3]      # [M, 3]
        
        # Compute pairwise distances
        distances = torch.cdist(pred_centers, gt_centers)  # [N, M]
        
        # Convert distance to similarity (inverse relationship)
        max_distance = 10.0  # Maximum reasonable distance
        similarities = torch.clamp(1.0 - distances / max_distance, min=0.0, max=1.0)
        
        return similarities
    
    def _store_predictions(self, predictions: Dict, batch: Dict, batch_idx: int):
        """Store predictions for detailed analysis"""
        # Store sample tokens and predictions
        if 'sample_token' in batch:
            for i, token in enumerate(batch['sample_token']):
                self.predictions.append({
                    'sample_token': token,
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'predictions': {k: v[i].cpu().numpy() if isinstance(v, torch.Tensor) else v[i] 
                                  for k, v in predictions.items()}
                })
    
    def _compute_final_metrics(self) -> Dict[str, float]:
        """Compute final validation metrics"""
        final_metrics = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                final_metrics[metric_name] = np.mean(values)
                final_metrics[f"{metric_name}_std"] = np.std(values)
        
        return final_metrics
    
    def run_nuscenes_evaluation(self, output_dir: str = "outputs/validation") -> Dict[str, Any]:
        """
        Run official nuScenes evaluation
        
        Args:
            output_dir: Directory to save evaluation results
            
        Returns:
            Evaluation results
        """
        logger.info("Running official nuScenes evaluation...")
        
        if not NUSCENES_AVAILABLE:
            logger.warning("nuScenes evaluation not available")
            return {}
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize nuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=str(self.data_root))
        
        # Generate predictions in nuScenes format
        predictions = self._generate_nuscenes_predictions(nusc)
        
        # Save predictions
        predictions_file = output_dir / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Run evaluation
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        
        eval_set = eval_set_map.get('v1.0-trainval', 'val')
        
        nusc_eval = NuScenesEval(
            nusc,
            config=config_factory('detection_cvpr_2019'),
            result_path=str(predictions_file),
            eval_set=eval_set,
            output_dir=str(output_dir),
            verbose=True
        )
        
        # Run evaluation
        metrics_summary = nusc_eval.main(plot_examples=10, render_curves=True)
        
        # Save results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info(f"nuScenes evaluation completed. Results saved to {output_dir}")
        return metrics_summary
    
    def _generate_nuscenes_predictions(self, nusc: NuScenes) -> List[Dict]:
        """Generate predictions in nuScenes format"""
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Generating predictions"):
                batch = self._move_batch_to_device(batch)
                model_predictions = self.model(batch)
                
                # Convert model predictions to nuScenes format
                batch_predictions = self._convert_to_nuscenes_format(
                    model_predictions, batch, nusc
                )
                predictions.extend(batch_predictions)
        
        return predictions
    
    def _convert_to_nuscenes_format(self, predictions: Dict, batch: Dict, nusc: NuScenes) -> List[Dict]:
        """Convert model predictions to nuScenes format"""
        # This is a simplified conversion - you'd need to implement proper conversion
        # based on your model's output format and nuScenes requirements
        
        nuScenes_predictions = []
        
        # Example conversion (adapt based on your model output)
        if 'pred_boxes' in predictions and 'sample_token' in batch:
            pred_boxes = predictions['pred_boxes']  # [B, N, 10]
            pred_scores = predictions.get('pred_scores', torch.ones(pred_boxes.size(0), pred_boxes.size(1)))
            pred_labels = predictions.get('pred_labels', torch.zeros(pred_boxes.size(0), pred_boxes.size(1), dtype=torch.long))
            
            for b in range(pred_boxes.size(0)):
                sample_token = batch['sample_token'][b]
                
                for n in range(pred_boxes.size(1)):
                    if pred_scores[b, n] > 0.1:  # Confidence threshold
                        prediction = {
                            'sample_token': sample_token,
                            'translation': pred_boxes[b, n, :3].cpu().numpy().tolist(),
                            'size': pred_boxes[b, n, 3:6].cpu().numpy().tolist(),
                            'rotation': pred_boxes[b, n, 6:10].cpu().numpy().tolist(),
                            'detection_name': 'car',  # Map to nuScenes class
                            'detection_score': pred_scores[b, n].item(),
                            'attribute_name': 'vehicle.parked'
                        }
                        nuScenes_predictions.append(prediction)
        
        return nuScenes_predictions
    
    def generate_visualizations(self, output_dir: str = "outputs/validation") -> None:
        """Generate validation visualizations"""
        logger.info("Generating validation visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot metrics over time
        self._plot_metrics(output_dir)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(output_dir)
        
        # Plot sample predictions
        self._plot_sample_predictions(output_dir)
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def _plot_metrics(self, output_dir: Path):
        """Plot validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Validation Metrics', fontsize=16)
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(self.metrics.items()):
            if values:
                ax = axes[i // 2, i % 2]
                ax.plot(values)
                ax.set_title(metric_name.replace('_', ' ').title())
                ax.set_xlabel('Batch')
                ax.set_ylabel('Value')
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, output_dir: Path):
        """Plot confusion matrix for classification"""
        # This would require proper label matching
        # For now, create a placeholder
        pass
    
    def _plot_sample_predictions(self, output_dir: Path):
        """Plot sample predictions vs ground truth"""
        # This would require proper visualization of 3D boxes
        # For now, create a placeholder
        pass


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='BEVNeXt-SAM2 Model Validation')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', default='data/nuscenes', help='Path to nuScenes dataset')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output-dir', default='outputs/validation', help='Output directory')
    parser.add_argument('--max-samples', type=int, help='Maximum samples to validate')
    parser.add_argument('--run-nuscenes-eval', action='store_true', help='Run official nuScenes evaluation')
    parser.add_argument('--generate-viz', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    try:
        # Create validator
        validator = ModelValidator(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            config=config
        )
        
        # Run validation
        metrics = validator.validate(max_samples=args.max_samples)
        
        # Print results
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        for metric_name, value in metrics.items():
            print(f"{metric_name:25s}: {value:.4f}")
        print("="*50)
        
        # Run nuScenes evaluation if requested
        if args.run_nuscenes_eval:
            nuscenes_results = validator.run_nuscenes_evaluation(args.output_dir)
            if nuscenes_results:
                print("\n" + "="*50)
                print("NUSCENES EVALUATION RESULTS")
                print("="*50)
                for metric_name, value in nuscenes_results.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric_name:25s}: {value:.4f}")
                print("="*50)
        
        # Generate visualizations if requested
        if args.generate_viz:
            validator.generate_visualizations(args.output_dir)
        
        # Save metrics
        metrics_file = Path(args.output_dir) / "validation_metrics.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nValidation completed! Results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main() 