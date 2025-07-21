#!/usr/bin/env python3
"""
nuScenes Dataset Loader for BEVNeXt-SAM2 Training

This module provides a proper dataset loader for nuScenes v1.0 full dataset
for training BEVNeXt-SAM2 models.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Any, Optional
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import random


class NuScenesTrainingDataset(Dataset):
    """nuScenes dataset for BEVNeXt-SAM2 training"""
    
    def __init__(self, 
                 data_root: str = "data/nuscenes",
                 version: str = "v1.0-trainval",
                 split: str = "train",
                 config: Dict = None):
        """
        Initialize nuScenes training dataset
        
        Args:
            data_root: Path to the nuScenes data directory  
            version: nuScenes version (e.g., 'v1.0-trainval', 'v1.0-mini')
            split: Dataset split ('train', 'val') 
            config: Training configuration dictionary
        """
        self.data_root = Path(data_root)
        self.version = version
        self.split = split
        self.config = config or {}
        
        # Initialize nuScenes API
        try:
            self.nusc = NuScenes(version=version, dataroot=str(self.data_root), verbose=True)
            print(f"nuScenes {version} loaded successfully")
        except Exception as e:
            print(f"Failed to load nuScenes {version}: {e}")
            print(f"Make sure the dataset is extracted to: {self.data_root}")
            raise
        
        # Camera names (6 cameras)
        self.camera_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
        ]
        
        # Class mapping for 10-class nuScenes detection
        self.class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        
        # Get scenes for the split
        self.scenes = self._get_scenes_for_split()
        
        # Get all samples for the split
        self.samples = self._get_samples_for_split()
        
        print(f"nuScenes {split} dataset initialized:")
        print(f"   └─ Scenes: {len(self.scenes)}")
        print(f"   └─ Samples: {len(self.samples)}")
        
        # Configuration 
        self.image_size = self.config.get('image_size', [224, 224])
        self.bev_size = self.config.get('bev_size', [48, 48])
        self.num_queries = self.config.get('num_queries', 200)
        
    def _get_scenes_for_split(self) -> List[str]:
        """Get scene tokens for the current split"""
        from nuscenes.utils import splits
        
        if self.version == 'v1.0-trainval':
            if self.split == 'train':
                scene_names = splits.train
            elif self.split == 'val':
                scene_names = splits.val
            else:
                raise ValueError(f"Unknown split: {self.split}")
        elif self.version == 'v1.0-mini':
            if self.split == 'train':
                scene_names = splits.mini_train
            elif self.split == 'val':
                scene_names = splits.mini_val
            else:
                raise ValueError(f"Unknown split: {self.split}")
        else:
            raise ValueError(f"Unknown version: {self.version}")
        
        # Convert scene names to tokens
        scene_tokens = []
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                scene_tokens.append(scene['token'])
                
        return scene_tokens
    
    def _get_samples_for_split(self) -> List[str]:
        """Get all sample tokens for the current split"""
        sample_tokens = []
        
        for scene_token in self.scenes:
            scene = self.nusc.get('scene', scene_token)
            
            # Get first sample in scene
            sample_token = scene['first_sample_token']
            
            # Traverse all samples in scene
            while sample_token != '':
                sample_tokens.append(sample_token)
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']
                
        return sample_tokens
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single training sample"""
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Load camera images
        camera_images, camera_intrinsics, camera_extrinsics = self._load_camera_data(sample)
        
        # Generate SAM2 masks (placeholder - in real implementation you'd run SAM2)
        sam2_masks = self._generate_sam2_masks(camera_images)
        
        # Load annotations
        gt_boxes, gt_labels, valid_mask = self._load_annotations(sample)
        
        # Prepare data for model
        return {
            'camera_images': torch.stack([torch.from_numpy(img.transpose(2, 0, 1)) for img in camera_images]).float() / 255.0,
            'sam2_masks': torch.stack([torch.from_numpy(mask) for mask in sam2_masks]).float(),
            'labels': gt_labels,
            'boxes': gt_boxes, 
            'valid_mask': valid_mask,
            'sample_token': sample_token,
            'camera_intrinsics': camera_intrinsics,
            'camera_extrinsics': camera_extrinsics
        }
    
    def _load_camera_data(self, sample: Dict) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
        """Load camera images and calibration data"""
        camera_images = []
        camera_intrinsics = []
        camera_extrinsics = []
        
        for cam_name in self.camera_names:
            # Get camera sample data
            camera_token = sample['data'][cam_name]
            camera_sample = self.nusc.get('sample_data', camera_token)
            
            # Load image
            image_path = self.data_root / camera_sample['filename']
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                image = image.resize(self.image_size)
                image = np.array(image)
            else:
                # Create placeholder if image not found
                image = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
                print(f"Warning: Image not found: {image_path}")
            
            camera_images.append(image)
            
            # Get calibration data
            calibrated_sensor_token = camera_sample['calibrated_sensor_token']
            calibrated_sensor = self.nusc.get('calibrated_sensor', calibrated_sensor_token)
            
            # Camera intrinsics
            camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
            camera_intrinsics.append(camera_intrinsic)
            
            # Camera extrinsics  
            rotation = np.array(calibrated_sensor['rotation'])
            translation = np.array(calibrated_sensor['translation'])
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = rotation
            extrinsic_matrix[:3, 3] = translation
            camera_extrinsics.append(extrinsic_matrix)
        
        return (camera_images, 
                torch.tensor(np.stack(camera_intrinsics), dtype=torch.float32),
                torch.tensor(np.stack(camera_extrinsics), dtype=torch.float32))
    
    def _generate_sam2_masks(self, camera_images: List[np.ndarray]) -> List[np.ndarray]:
        """Generate SAM2 masks (placeholder - replace with actual SAM2 inference)"""
        sam2_masks = []
        
        for image in camera_images:
            H, W = image.shape[:2]
            
            # For now, generate random masks as placeholder
            # In real implementation, you would run SAM2 here
            mask = np.random.randint(0, 2, (1, H, W), dtype=np.float32)
            sam2_masks.append(mask)
            
        return sam2_masks
    
    def _load_annotations(self, sample: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load 3D bounding box annotations"""
        # Get all annotations for this sample
        annotations = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            annotations.append(ann)
        
        # Filter annotations to only include relevant classes
        filtered_annotations = []
        for ann in annotations:
            category_name = ann['category_name']
            # Map nuScenes categories to our class names
            if self._map_category_to_class(category_name) is not None:
                filtered_annotations.append(ann)
        
        # Pad or truncate to num_queries
        num_annotations = len(filtered_annotations)
        max_annotations = self.num_queries
        
        gt_boxes = np.zeros((max_annotations, 10), dtype=np.float32)  # x,y,z,w,l,h,rot,vx,vy,vz
        gt_labels = np.zeros(max_annotations, dtype=np.int64)
        valid_mask = np.zeros(max_annotations, dtype=bool)
        
        for i, ann in enumerate(filtered_annotations[:max_annotations]):
            # Get box parameters
            translation = ann['translation']  # x, y, z
            size = ann['size']  # w, l, h
            rotation = ann['rotation']  # quaternion
            
            # Convert quaternion to yaw angle 
            from pyquaternion import Quaternion
            q = Quaternion(rotation)
            yaw = q.yaw_pitch_roll[0]
            
            # Velocity (if available)
            velocity = ann.get('velocity', [0.0, 0.0])
            
            # Fill box parameters
            gt_boxes[i] = [
                translation[0], translation[1], translation[2],  # x, y, z
                size[0], size[1], size[2],  # w, l, h  
                yaw,  # rotation
                velocity[0], velocity[1], 0.0  # vx, vy, vz
            ]
            
            # Get class label
            class_idx = self._map_category_to_class(ann['category_name'])
            gt_labels[i] = class_idx if class_idx is not None else 0
            
            valid_mask[i] = True
        
        return (torch.tensor(gt_boxes, dtype=torch.float32),
                torch.tensor(gt_labels, dtype=torch.long), 
                torch.tensor(valid_mask, dtype=torch.bool))
    
    def _map_category_to_class(self, category_name: str) -> Optional[int]:
        """Map nuScenes category to class index"""
        # Simplified mapping - you may want to make this more sophisticated
        category_mapping = {
            'vehicle.car': 0,
            'vehicle.truck': 1, 
            'vehicle.construction': 2,
            'vehicle.bus.bendy': 3,
            'vehicle.bus.rigid': 3,
            'vehicle.trailer': 4,
            'movable_object.barrier': 5,
            'vehicle.motorcycle': 6,
            'vehicle.bicycle': 7,
            'human.pedestrian.adult': 8,
            'human.pedestrian.child': 8,
            'movable_object.trafficcone': 9
        }
        
        return category_mapping.get(category_name, None)


def create_nuscenes_dataloader(data_root: str = "data/nuscenes",
                              version: str = "v1.0-trainval", 
                              split: str = "train",
                              config: Dict = None,
                              batch_size: int = 1,
                              num_workers: int = 2,
                              shuffle: bool = True) -> torch.utils.data.DataLoader:
    """Create nuScenes data loader for training"""
    
    dataset = NuScenesTrainingDataset(
        data_root=data_root,
        version=version, 
        split=split,
        config=config
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: {
            'camera_images': torch.stack([item['camera_images'] for item in batch]),
            'sam2_masks': torch.stack([item['sam2_masks'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'boxes': torch.stack([item['boxes'] for item in batch]),
            'valid_mask': torch.stack([item['valid_mask'] for item in batch]),
            'sample_tokens': [item['sample_token'] for item in batch],
            'camera_intrinsics': torch.stack([item['camera_intrinsics'] for item in batch]),
            'camera_extrinsics': torch.stack([item['camera_extrinsics'] for item in batch])
        }
    ) 