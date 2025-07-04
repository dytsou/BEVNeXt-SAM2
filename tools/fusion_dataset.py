"""
Fusion Dataset for BEVNeXt-SAM2 Training

Dataset class that provides data for joint 3D detection and 2D segmentation training.
Handles multi-view images, 3D annotations, and segmentation masks.
"""

import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A

from mmcv import Config
from pyquaternion import Quaternion


class FusionDataset(Dataset):
    """
    Dataset for BEVNeXt-SAM2 fusion training.
    
    Provides:
    - Multi-view camera images
    - 3D bounding box annotations
    - 2D segmentation masks (if available)
    - Camera calibration data
    - Metadata for coordinate transformations
    """
    
    def __init__(
        self,
        data_config: Union[str, Dict],
        split: str = 'train',
        transform: Optional[List] = None,
        load_masks: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize fusion dataset.
        
        Args:
            data_config: Dataset configuration file or dict
            split: Dataset split ('train', 'val', 'test')
            transform: Data augmentation transforms
            load_masks: Whether to load segmentation masks
            max_samples: Maximum number of samples (for debugging)
        """
        super().__init__()
        
        self.split = split
        self.transform = transform
        self.load_masks = load_masks
        
        # Load configuration
        if isinstance(data_config, str):
            self.cfg = Config.fromfile(data_config)
        else:
            self.cfg = data_config
            
        # Setup data paths
        self.data_root = Path(self.cfg.data_root)
        self.annotation_file = self.data_root / self.cfg.splits[split]['ann_file']
        
        # Load annotations
        self.load_annotations()
        
        # Limit samples if specified
        if max_samples is not None:
            self.data_infos = self.data_infos[:max_samples]
            
        # Setup transforms
        self.setup_transforms()
        
        print(f"Loaded {len(self.data_infos)} {split} samples")
        
    def load_annotations(self):
        """Load dataset annotations."""
        print(f"Loading annotations from {self.annotation_file}")
        
        if self.annotation_file.suffix == '.pkl':
            with open(self.annotation_file, 'rb') as f:
                self.data_infos = pickle.load(f)
        elif self.annotation_file.suffix == '.json':
            import json
            with open(self.annotation_file, 'r') as f:
                self.data_infos = json.load(f)
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_file.suffix}")
        
        # Filter valid samples
        self.data_infos = [info for info in self.data_infos if self.is_valid_sample(info)]
        
    def is_valid_sample(self, info: Dict) -> bool:
        """Check if a sample is valid."""
        # Check if all required camera images exist
        for cam_name in self.cfg.camera_names:
            img_path = self.data_root / info['cams'][cam_name]['data_path']
            if not img_path.exists():
                return False
                
        # Check if segmentation masks exist (if required)
        if self.load_masks and 'masks' in info:
            for cam_name in self.cfg.camera_names:
                if cam_name in info['masks']:
                    mask_path = self.data_root / info['masks'][cam_name]
                    if not mask_path.exists():
                        return False
                        
        return True
        
    def setup_transforms(self):
        """Setup data augmentation transforms."""
        if self.transform is None:
            if self.split == 'train':
                self.transform = self.get_default_train_transforms()
            else:
                self.transform = self.get_default_val_transforms()
                
    def get_default_train_transforms(self):
        """Get default training transforms."""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(90, 110), p=0.3),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.MotionBlur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def get_default_val_transforms(self):
        """Get default validation transforms."""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self) -> int:
        return len(self.data_infos)
        
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        info = self.data_infos[idx]
        
        # Load multi-view images
        images, img_metas = self.load_images(info)
        
        # Load 3D annotations
        gt_bboxes_3d, gt_labels_3d = self.load_3d_annotations(info)
        
        # Load segmentation masks (if available)
        gt_masks = None
        if self.load_masks and 'masks' in info:
            gt_masks = self.load_masks_annotations(info)
        
        # Apply transforms
        if self.transform is not None:
            images, gt_masks = self.apply_transforms(images, gt_masks)
        
        # Convert to tensors
        sample = {
            'images': torch.from_numpy(images).float(),
            'img_metas': img_metas,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d,
            'sample_idx': idx,
            'token': info.get('token', f'sample_{idx}')
        }
        
        if gt_masks is not None:
            sample['gt_masks'] = gt_masks
            
        return sample
        
    def load_images(self, info: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Load multi-view camera images."""
        images = []
        img_metas = []
        
        for cam_name in self.cfg.camera_names:
            cam_info = info['cams'][cam_name]
            
            # Load image
            img_path = self.data_root / cam_info['data_path']
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if hasattr(self.cfg, 'img_size'):
                image = cv2.resize(image, self.cfg.img_size)
            
            images.append(image)
            
            # Create image meta
            img_meta = {
                'filename': str(img_path),
                'cam_name': cam_name,
                'img_shape': image.shape,
                'cam_intrinsic': np.array(cam_info['cam_intrinsic']),
                'sensor2lidar_rotation': np.array(cam_info['sensor2lidar_rotation']),
                'sensor2lidar_translation': np.array(cam_info['sensor2lidar_translation']),
                'lidar2ego_rotation': np.array(info.get('lidar2ego_rotation', np.eye(3))),
                'lidar2ego_translation': np.array(info.get('lidar2ego_translation', [0, 0, 0])),
                'ego2global_rotation': np.array(info.get('ego2global_rotation', np.eye(3))),
                'ego2global_translation': np.array(info.get('ego2global_translation', [0, 0, 0])),
                'timestamp': cam_info.get('timestamp', 0)
            }
            img_metas.append(img_meta)
        
        # Stack images: [N, H, W, C] where N is number of views
        images = np.stack(images, axis=0)
        
        return images, img_metas
        
    def load_3d_annotations(self, info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load 3D bounding box annotations."""
        if 'gt_boxes' not in info:
            # Return empty annotations
            return torch.zeros(0, 7), torch.zeros(0, dtype=torch.long)
        
        gt_boxes = np.array(info['gt_boxes'])  # [N, 7] - x, y, z, w, l, h, yaw
        gt_names = info['gt_names']
        
        # Convert names to label indices
        gt_labels = []
        for name in gt_names:
            if name in self.cfg.class_names:
                gt_labels.append(self.cfg.class_names.index(name))
            else:
                gt_labels.append(-1)  # Unknown class
        
        gt_labels = np.array(gt_labels)
        
        # Filter out unknown classes
        valid_mask = gt_labels >= 0
        gt_boxes = gt_boxes[valid_mask]
        gt_labels = gt_labels[valid_mask]
        
        return torch.from_numpy(gt_boxes).float(), torch.from_numpy(gt_labels).long()
        
    def load_masks_annotations(self, info: Dict) -> List[torch.Tensor]:
        """Load segmentation mask annotations."""
        masks = []
        
        for cam_name in self.cfg.camera_names:
            if cam_name in info['masks']:
                mask_path = self.data_root / info['masks'][cam_name]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # Resize if needed
                if hasattr(self.cfg, 'img_size'):
                    mask = cv2.resize(mask, self.cfg.img_size)
                
                # Convert to binary mask
                mask = (mask > 0).astype(np.uint8)
                masks.append(torch.from_numpy(mask).float())
            else:
                # Create empty mask if not available
                h, w = self.cfg.img_size if hasattr(self.cfg, 'img_size') else (900, 1600)
                masks.append(torch.zeros(h, w, dtype=torch.float))
        
        return masks
        
    def apply_transforms(
        self,
        images: np.ndarray,
        masks: Optional[List[torch.Tensor]] = None
    ) -> Tuple[np.ndarray, Optional[List[torch.Tensor]]]:
        """Apply data augmentation transforms."""
        if self.transform is None:
            # Convert to CHW format
            images = images.transpose(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
            return images, masks
        
        transformed_images = []
        transformed_masks = []
        
        for i in range(images.shape[0]):
            image = images[i]
            mask = masks[i].numpy() if masks is not None else None
            
            # Apply transforms
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                transformed_images.append(transformed['image'])
                transformed_masks.append(torch.from_numpy(transformed['mask']).float())
            else:
                transformed = self.transform(image=image)
                transformed_images.append(transformed['image'])
        
        # Stack and convert to CHW format
        images = np.stack(transformed_images, axis=0)
        images = images.transpose(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        
        if masks is not None:
            masks = transformed_masks
        
        return images, masks


def create_fusion_dataloader(
    dataset: FusionDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """Create a data loader for the fusion dataset."""
    
    def collate_fn(batch):
        """Custom collate function for fusion data."""
        # Separate different data types
        images = torch.stack([item['images'] for item in batch])
        img_metas = [item['img_metas'] for item in batch]
        gt_bboxes_3d = [item['gt_bboxes_3d'] for item in batch]
        gt_labels_3d = [item['gt_labels_3d'] for item in batch]
        sample_idx = [item['sample_idx'] for item in batch]
        tokens = [item['token'] for item in batch]
        
        # Handle optional masks
        gt_masks = None
        if 'gt_masks' in batch[0]:
            gt_masks = [item['gt_masks'] for item in batch]
        
        return {
            'images': images,
            'img_metas': img_metas,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d,
            'gt_masks': gt_masks,
            'sample_idx': sample_idx,
            'tokens': tokens
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


class SyntheticFusionDataset(FusionDataset):
    """
    Synthetic dataset for testing the fusion pipeline without real data.
    Generates dummy multi-view images, 3D boxes, and segmentation masks.
    """
    
    def __init__(self, num_samples: int = 100, **kwargs):
        self.num_samples = num_samples
        self.cfg = self.create_dummy_config()
        self.split = kwargs.get('split', 'train')
        self.transform = kwargs.get('transform', None)
        self.load_masks = kwargs.get('load_masks', True)
        
        # Create dummy data infos
        self.data_infos = [self.create_dummy_info(i) for i in range(num_samples)]
        
        # Setup transforms
        self.setup_transforms()
        
        print(f"Created synthetic dataset with {num_samples} samples")
        
    def create_dummy_config(self):
        """Create dummy configuration for synthetic data."""
        from types import SimpleNamespace
        
        cfg = SimpleNamespace()
        cfg.camera_names = ['front', 'front_left', 'front_right', 'back', 'back_left', 'back_right']
        cfg.img_size = (448, 800)  # H, W
        cfg.class_names = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']
        cfg.num_classes = len(cfg.class_names)
        
        return cfg
        
    def create_dummy_info(self, idx: int) -> Dict:
        """Create dummy sample info."""
        return {
            'token': f'dummy_{idx}',
            'cams': {cam: {'dummy': True} for cam in self.cfg.camera_names},
            'gt_boxes': np.random.rand(random.randint(1, 10), 7) * 10,  # Random 3D boxes
            'gt_names': random.choices(self.cfg.class_names, k=random.randint(1, 10)),
            'masks': {cam: f'dummy_mask_{cam}' for cam in self.cfg.camera_names} if self.load_masks else {}
        }
        
    def load_images(self, info: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Generate dummy multi-view images."""
        images = []
        img_metas = []
        
        H, W = self.cfg.img_size
        
        for cam_name in self.cfg.camera_names:
            # Generate random image
            image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
            images.append(image)
            
            # Create dummy camera parameters
            img_meta = {
                'filename': f'dummy_{cam_name}.jpg',
                'cam_name': cam_name,
                'img_shape': (H, W, 3),
                'cam_intrinsic': np.array([[800, 0, W/2], [0, 800, H/2], [0, 0, 1]]),
                'sensor2lidar_rotation': np.eye(3),
                'sensor2lidar_translation': np.array([0, 0, 0]),
                'lidar2ego_rotation': np.eye(3),
                'lidar2ego_translation': np.array([0, 0, 0]),
                'ego2global_rotation': np.eye(3),
                'ego2global_translation': np.array([0, 0, 0]),
                'timestamp': 0
            }
            img_metas.append(img_meta)
        
        images = np.stack(images, axis=0)
        return images, img_metas
        
    def load_masks_annotations(self, info: Dict) -> List[torch.Tensor]:
        """Generate dummy segmentation masks."""
        masks = []
        H, W = self.cfg.img_size
        
        for cam_name in self.cfg.camera_names:
            # Generate random binary mask
            mask = np.random.choice([0, 1], size=(H, W), p=[0.8, 0.2])
            masks.append(torch.from_numpy(mask.astype(np.float32)))
        
        return masks
        
    def is_valid_sample(self, info: Dict) -> bool:
        """All synthetic samples are valid."""
        return True