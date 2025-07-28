#!/usr/bin/env python3
"""
Comprehensive nuScenes v1.0 Dataset Loader for BEVNeXt-SAM2 Training

This module provides a professional-grade dataset loader specifically designed for nuScenes v1.0
dataset characteristics including token-based associations, multi-modal sensor data, and 
comprehensive 3D annotation processing.

Author: Senior Python Programmer & Autonomous Driving Dataset Expert
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Any, Optional, Union
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.splits import create_splits_scenes
import random
import logging
from dataclasses import dataclass
from collections import defaultdict
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NuScenesConfig:
    """Configuration class for nuScenes dataset parameters"""
    # Data paths
    data_root: str = "data/nuscenes"
    version: str = "v1.0-trainval"
    
    # Sensor configuration
    camera_names: List[str] = None
    use_lidar: bool = True
    use_radar: bool = False
    use_camera: bool = True
    
    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    image_mean: List[float] = None
    image_std: List[float] = None
    
    # BEV configuration  
    bev_size: Tuple[int, int] = (48, 48)
    point_cloud_range: List[float] = None
    
    # Training parameters
    num_queries: int = 200
    max_sweeps: int = 10
    
    # Data augmentation
    use_augmentation: bool = True
    flip_ratio: float = 0.5
    rotation_range: Tuple[float, float] = (-0.3925, 0.3925)
    scale_range: Tuple[float, float] = (0.95, 1.05)
    
    def __post_init__(self):
        """Initialize default values after instantiation"""
        if self.camera_names is None:
            self.camera_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
            ]
        if self.image_mean is None:
            self.image_mean = [0.485, 0.456, 0.406]  # ImageNet defaults
        if self.image_std is None:
            self.image_std = [0.229, 0.224, 0.225]   # ImageNet defaults
        if self.point_cloud_range is None:
            self.point_cloud_range = [-50, -50, -5, 50, 50, 3]


class NuScenesDataAnalyzer:
    """Utility class for nuScenes dataset analysis and statistics"""
    
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics"""
        stats = {
            'total_scenes': len(self.nusc.scene),
            'total_samples': len(self.nusc.sample),
            'total_annotations': len(self.nusc.sample_annotation),
            'total_sample_data': len(self.nusc.sample_data),
            'sensor_statistics': self._get_sensor_statistics(),
            'category_statistics': self._get_category_statistics(),
            'scene_statistics': self._get_scene_statistics(),
            'temporal_statistics': self._get_temporal_statistics()
        }
        return stats
    
    def _get_sensor_statistics(self) -> Dict[str, int]:
        """Get statistics for each sensor type"""
        sensor_counts = defaultdict(int)
        for sample_data in self.nusc.sample_data:
            sensor_token = sample_data['sensor_token']
            sensor = self.nusc.get('sensor', sensor_token)
            sensor_counts[sensor['channel']] += 1
        return dict(sensor_counts)
    
    def _get_category_statistics(self) -> Dict[str, int]:
        """Get statistics for each object category"""
        category_counts = defaultdict(int)
        for annotation in self.nusc.sample_annotation:
            category_counts[annotation['category_name']] += 1
        return dict(category_counts)
    
    def _get_scene_statistics(self) -> Dict[str, Any]:
        """Get scene-level statistics"""
        scene_lengths = []
        location_counts = defaultdict(int)
        
        for scene in self.nusc.scene:
            scene_lengths.append(scene['nbr_samples'])
            log = self.nusc.get('log', scene['log_token'])
            location_counts[log['location']] += 1
            
        return {
            'avg_scene_length': np.mean(scene_lengths),
            'min_scene_length': np.min(scene_lengths),
            'max_scene_length': np.max(scene_lengths),
            'location_distribution': dict(location_counts)
        }
    
    def _get_temporal_statistics(self) -> Dict[str, float]:
        """Get temporal statistics"""
        timestamps = [sample['timestamp'] for sample in self.nusc.sample]
        timestamps.sort()
        
        intervals = np.diff(timestamps) / 1e6  # Convert to seconds
        
        return {
            'total_duration_hours': (timestamps[-1] - timestamps[0]) / 1e6 / 3600,
            'avg_sample_interval_sec': np.mean(intervals),
            'min_sample_interval_sec': np.min(intervals),
            'max_sample_interval_sec': np.max(intervals)
        }


class NuScenesTokenManager:
    """Manages token-based associations in nuScenes dataset"""
    
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self._build_token_maps()
        
    def _build_token_maps(self):
        """Build fast lookup maps for token-based associations"""
        logger.info("Building token association maps...")
        
        # Sample token to scene token mapping
        self.sample_to_scene = {}
        for scene in self.nusc.scene:
            sample_token = scene['first_sample_token']
            while sample_token != '':
                sample = self.nusc.get('sample', sample_token)
                self.sample_to_scene[sample_token] = scene['token']
                sample_token = sample['next']
        
        # Sample token to annotations mapping
        self.sample_to_annotations = defaultdict(list)
        for annotation in self.nusc.sample_annotation:
            self.sample_to_annotations[annotation['sample_token']].append(annotation['token'])
        
        # Sample token to sample_data mapping
        self.sample_to_sample_data = defaultdict(dict)
        for sample_data in self.nusc.sample_data:
            if sample_data['is_key_frame']:
                sample_token = sample_data['sample_token']
                sensor_token = sample_data['sensor_token']
                sensor = self.nusc.get('sensor', sensor_token)
                self.sample_to_sample_data[sample_token][sensor['channel']] = sample_data['token']
        
        logger.info(f"Built token maps for {len(self.sample_to_scene)} samples")
    
    def get_scene_samples(self, scene_token: str) -> List[str]:
        """Get all sample tokens for a given scene"""
        scene = self.nusc.get('scene', scene_token)
        samples = []
        sample_token = scene['first_sample_token']
        
        while sample_token != '':
            samples.append(sample_token)
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
            
        return samples
    
    def get_sample_sensor_data(self, sample_token: str, sensor_channel: str) -> Optional[str]:
        """Get sample_data token for specific sensor in a sample"""
        return self.sample_to_sample_data.get(sample_token, {}).get(sensor_channel)
    
    def validate_token_integrity(self) -> Dict[str, Any]:
        """Validate integrity of token associations"""
        issues = {
            'missing_sensor_data': [],
            'broken_sample_chains': [],
            'orphaned_annotations': [],
            'missing_files': []
        }
        
        # Check for missing sensor data
        for sample_token in self.sample_to_scene.keys():
            sample_data = self.sample_to_sample_data.get(sample_token, {})
            for camera in ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                if camera not in sample_data:
                    issues['missing_sensor_data'].append((sample_token, camera))
        
        # Check file existence
        for sample_data in self.nusc.sample_data:
            if sample_data['is_key_frame']:
                file_path = Path(self.nusc.dataroot) / sample_data['filename']
                if not file_path.exists():
                    issues['missing_files'].append(sample_data['filename'])
        
        return issues


class NuScenesMultiModalDataset(Dataset):
    """
    Comprehensive nuScenes dataset for multi-modal autonomous driving data.
    
    Handles token-based associations, multi-modal sensor data (camera, LiDAR, radar),
    and supports both key frames and sweeps for temporal modeling.
    """
    
    # nuScenes 23 object categories with mapping to simplified classes
    NUSCENES_CATEGORIES = {
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
        'human.pedestrian.wheelchair': 8,
        'human.pedestrian.stroller': 8,
        'human.pedestrian.personal_mobility': 8,
        'human.pedestrian.police_officer': 8,
        'human.pedestrian.construction_worker': 8,
        'movable_object.trafficcone': 9,
        'movable_object.pushable_pullable': 5,
        'movable_object.debris': 5,
        'static_object.bicycle_rack': 5,
        'animal': 8,  # Treat as pedestrian
        'vehicle.emergency.ambulance': 1,  # Treat as truck
        'vehicle.emergency.police': 0  # Treat as car
    }
    
    SIMPLIFIED_CLASSES = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    
    def __init__(self, 
                 config: NuScenesConfig,
                 split: str = "train",
                 verbose: bool = True):
        """
        Initialize nuScenes multi-modal dataset
        
        Args:
            config: NuScenesConfig object with dataset parameters
            split: Dataset split ('train', 'val', 'test', 'mini_train', 'mini_val')
            verbose: Enable verbose logging
        """
        self.config = config
        self.split = split
        self.verbose = verbose
        
        # Initialize nuScenes API
        try:
            self.nusc = NuScenes(
                version=config.version, 
                dataroot=str(config.data_root), 
                verbose=verbose
            )
            logger.info(f"nuScenes {config.version} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load nuScenes {config.version}: {e}")
            raise
        
        # Initialize token manager and analyzer
        self.token_manager = NuScenesTokenManager(self.nusc)
        self.analyzer = NuScenesDataAnalyzer(self.nusc)
        
        # Get dataset statistics
        if verbose:
            stats = self.analyzer.get_dataset_statistics()
            logger.info(f"Dataset Statistics:")
            logger.info(f"   Total scenes: {stats['total_scenes']}")
            logger.info(f"   Total samples: {stats['total_samples']}")
            logger.info(f"   Total annotations: {stats['total_annotations']}")
        
        # Get samples for the specified split
        self.scene_tokens = self._get_scene_tokens_for_split()
        self.sample_tokens = self._get_sample_tokens_for_split()
        
        logger.info(f"nuScenes {split} dataset initialized:")
        logger.info(f"   └─ Scenes: {len(self.scene_tokens)}")
        logger.info(f"   └─ Samples: {len(self.sample_tokens)}")
        
        # Initialize data transformations
        self._setup_transforms()
        
    def _get_scene_tokens_for_split(self) -> List[str]:
        """Get scene tokens for the specified split"""
        # Use nuScenes official splits
        splits = create_splits_scenes()
        
        if self.config.version == 'v1.0-trainval':
            if self.split == 'train':
                scene_names = splits['train']
            elif self.split == 'val':
                scene_names = splits['val']
            else:
                raise ValueError(f"Unknown split: {self.split} for version {self.config.version}")
        elif self.config.version == 'v1.0-mini':
            if self.split in ['train', 'mini_train']:
                scene_names = splits['mini_train']
            elif self.split in ['val', 'mini_val']:
                scene_names = splits['mini_val']
            else:
                raise ValueError(f"Unknown split: {self.split} for version {self.config.version}")
        else:
            raise ValueError(f"Unsupported version: {self.config.version}")
        
        # Convert scene names to tokens
        scene_tokens = []
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                scene_tokens.append(scene['token'])
                
        return scene_tokens
    
    def _get_sample_tokens_for_split(self) -> List[str]:
        """Get all sample tokens for the specified split"""
        sample_tokens = []
        for scene_token in self.scene_tokens:
            scene_samples = self.token_manager.get_scene_samples(scene_token)
            sample_tokens.extend(scene_samples)
        return sample_tokens
    
    def _setup_transforms(self):
        """Setup data transformations for images"""
        import torchvision.transforms as transforms
        
        # Image normalization
        self.image_transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.image_mean,
                std=self.config.image_std
            )
        ])
        
        # Augmentation transforms for training
        if self.config.use_augmentation and self.split == 'train':
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=self.config.flip_ratio)
            ])
        else:
            self.augment_transform = None
    
    def __len__(self) -> int:
        return len(self.sample_tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a complete multi-modal sample from nuScenes dataset
        
        Returns:
            Dictionary containing:
            - camera_images: Tensor [N_cams, 3, H, W]
            - lidar_points: Tensor [N_points, 4] (x, y, z, intensity)
            - radar_points: Tensor [N_points, 6] (x, y, z, vx, vy, rcs)
            - gt_boxes: Tensor [N_objects, 10] (x, y, z, w, l, h, yaw, vx, vy, vz)
            - gt_labels: Tensor [N_objects]
            - gt_valid: Tensor [N_objects] (boolean mask)
            - sample_token: str
            - metadata: Dict with additional information
        """
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Load multi-modal sensor data
        camera_data = self._load_camera_data(sample_token) if self.config.use_camera else None
        lidar_data = self._load_lidar_data(sample_token) if self.config.use_lidar else None
        radar_data = self._load_radar_data(sample_token) if self.config.use_radar else None
        
        # Load annotations
        annotations_data = self._load_annotations(sample_token)
        
        # Prepare return dictionary
        sample_dict = {
            'sample_token': sample_token,
            'timestamp': sample['timestamp'],
            'scene_token': self.token_manager.sample_to_scene[sample_token],
            **annotations_data
        }
        
        # Add sensor data
        if camera_data is not None:
            sample_dict.update(camera_data)
        if lidar_data is not None:
            sample_dict.update(lidar_data)
        if radar_data is not None:
            sample_dict.update(radar_data)
        
        # Add metadata
        sample_dict['metadata'] = self._get_sample_metadata(sample_token)
        
        return sample_dict
    
    def _load_camera_data(self, sample_token: str) -> Dict[str, torch.Tensor]:
        """Load multi-camera data for a sample"""
        camera_images = []
        camera_intrinsics = []
        camera_extrinsics = []
        
        for camera_name in self.config.camera_names:
            # Get sample_data token for this camera
            sample_data_token = self.token_manager.get_sample_sensor_data(sample_token, camera_name)
            
            if sample_data_token is None:
                # Create placeholder if camera data missing
                placeholder_image = torch.zeros(3, *self.config.image_size)
                placeholder_intrinsic = torch.eye(3)
                placeholder_extrinsic = torch.eye(4)
                
                camera_images.append(placeholder_image)
                camera_intrinsics.append(placeholder_intrinsic)
                camera_extrinsics.append(placeholder_extrinsic)
                continue
            
            sample_data = self.nusc.get('sample_data', sample_data_token)
            
            # Load image
            image_path = Path(self.nusc.dataroot) / sample_data['filename']
            try:
                image = Image.open(image_path).convert('RGB')
                
                # Apply augmentations if training
                if self.augment_transform is not None:
                    image = self.augment_transform(image)
                
                # Apply standard transforms
                image = self.image_transform(image)
                camera_images.append(image)
                
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                placeholder_image = torch.zeros(3, *self.config.image_size)
                camera_images.append(placeholder_image)
            
            # Get camera calibration
            calibrated_sensor = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            
            # Camera intrinsics
            intrinsic = torch.tensor(calibrated_sensor['camera_intrinsic'], dtype=torch.float32)
            camera_intrinsics.append(intrinsic)
            
            # Camera extrinsics (sensor to ego transformation)
            extrinsic = torch.tensor(transform_matrix(
                calibrated_sensor['translation'],
                calibrated_sensor['rotation']
            ), dtype=torch.float32)
            camera_extrinsics.append(extrinsic)
        
        return {
            'camera_images': torch.stack(camera_images),
            'camera_intrinsics': torch.stack(camera_intrinsics),
            'camera_extrinsics': torch.stack(camera_extrinsics)
        }
    
    def _load_lidar_data(self, sample_token: str) -> Dict[str, torch.Tensor]:
        """Load LiDAR point cloud data for a sample"""
        sample_data_token = self.token_manager.get_sample_sensor_data(sample_token, 'LIDAR_TOP')
        
        if sample_data_token is None:
            return {'lidar_points': torch.zeros(0, 4)}
        
        sample_data = self.nusc.get('sample_data', sample_data_token)
        
        try:
            # Load LiDAR point cloud
            lidar_path = Path(self.nusc.dataroot) / sample_data['filename']
            pc = LidarPointCloud.from_file(str(lidar_path))
            
            # Convert to tensor [N, 4] (x, y, z, intensity)
            points = torch.tensor(pc.points[:4].T, dtype=torch.float32)
            
            # Filter points within point cloud range
            if self.config.point_cloud_range is not None:
                x_min, y_min, z_min, x_max, y_max, z_max = self.config.point_cloud_range
                mask = (
                    (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                    (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                )
                points = points[mask]
            
            return {'lidar_points': points}
            
        except Exception as e:
            logger.warning(f"Failed to load LiDAR data: {e}")
            return {'lidar_points': torch.zeros(0, 4)}
    
    def _load_radar_data(self, sample_token: str) -> Dict[str, torch.Tensor]:
        """Load radar data for a sample"""
        radar_channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        all_radar_points = []
        
        for radar_name in radar_channels:
            sample_data_token = self.token_manager.get_sample_sensor_data(sample_token, radar_name)
            
            if sample_data_token is None:
                continue
                
            sample_data = self.nusc.get('sample_data', sample_data_token)
            
            try:
                # Load radar point cloud
                radar_path = Path(self.nusc.dataroot) / sample_data['filename']
                pc = RadarPointCloud.from_file(str(radar_path))
                
                # Convert to tensor [N, 6] (x, y, z, vx, vy, rcs)
                points = torch.tensor(pc.points.T, dtype=torch.float32)
                all_radar_points.append(points)
                
            except Exception as e:
                logger.warning(f"Failed to load radar data from {radar_name}: {e}")
                continue
        
        if all_radar_points:
            radar_points = torch.cat(all_radar_points, dim=0)
        else:
            radar_points = torch.zeros(0, 6)
        
        return {'radar_points': radar_points}
    
    def _load_annotations(self, sample_token: str) -> Dict[str, torch.Tensor]:
        """Load 3D bounding box annotations for a sample"""
        annotation_tokens = self.token_manager.sample_to_annotations.get(sample_token, [])
        
        if not annotation_tokens:
            return {
                'gt_boxes': torch.zeros(0, 10),
                'gt_labels': torch.zeros(0, dtype=torch.long),
                'gt_valid': torch.zeros(0, dtype=torch.bool)
            }
        
        boxes = []
        labels = []
        valid_mask = []
        
        for ann_token in annotation_tokens:
            annotation = self.nusc.get('sample_annotation', ann_token)
            
            # Map category to simplified class
            category_name = annotation['category_name']
            class_idx = self.NUSCENES_CATEGORIES.get(category_name)
            
            if class_idx is None:
                continue  # Skip unknown categories
            
            # Get 3D bounding box parameters
            translation = annotation['translation']  # [x, y, z]
            size = annotation['size']  # [w, l, h]
            rotation = annotation['rotation']  # quaternion [w, x, y, z]
            
            # Convert quaternion to yaw angle
            from pyquaternion import Quaternion
            q = Quaternion(rotation)
            yaw = q.yaw_pitch_roll[0]
            
            # Get velocity if available
            velocity = annotation.get('velocity', [0.0, 0.0])
            
            # Create bounding box [x, y, z, w, l, h, yaw, vx, vy, vz]
            box = [
                translation[0], translation[1], translation[2],  # position
                size[0], size[1], size[2],  # dimensions
                yaw,  # rotation
                velocity[0], velocity[1], 0.0  # velocity
            ]
            
            boxes.append(box)
            labels.append(class_idx)
            valid_mask.append(True)
        
        # Pad or truncate to num_queries
        num_objects = len(boxes)
        max_objects = self.config.num_queries
        
        gt_boxes = torch.zeros(max_objects, 10, dtype=torch.float32)
        gt_labels = torch.zeros(max_objects, dtype=torch.long)
        gt_valid = torch.zeros(max_objects, dtype=torch.bool)
        
        if num_objects > 0:
            num_to_use = min(num_objects, max_objects)
            gt_boxes[:num_to_use] = torch.tensor(boxes[:num_to_use], dtype=torch.float32)
            gt_labels[:num_to_use] = torch.tensor(labels[:num_to_use], dtype=torch.long)
            gt_valid[:num_to_use] = True
        
        return {
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'gt_valid': gt_valid
        }
    
    def _get_sample_metadata(self, sample_token: str) -> Dict[str, Any]:
        """Get additional metadata for a sample"""
        sample = self.nusc.get('sample', sample_token)
        scene_token = self.token_manager.sample_to_scene[sample_token]
        scene = self.nusc.get('scene', scene_token)
        log = self.nusc.get('log', scene['log_token'])
        
        return {
            'scene_name': scene['name'],
            'scene_description': scene['description'],
            'location': log['location'],
            'vehicle': log['vehicle'],
            'date_captured': log['date_captured'],
            'sample_idx_in_scene': self._get_sample_index_in_scene(sample_token, scene_token)
        }
    
    def _get_sample_index_in_scene(self, sample_token: str, scene_token: str) -> int:
        """Get the index of a sample within its scene"""
        scene_samples = self.token_manager.get_scene_samples(scene_token)
        return scene_samples.index(sample_token)


def create_nuscenes_dataloader(
    config: NuScenesConfig,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for nuScenes multi-modal dataset
    
    Args:
        config: NuScenesConfig object with dataset parameters
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader object
    """
    dataset = NuScenesMultiModalDataset(config=config, split=split)
    
    def collate_fn(batch):
        """Custom collate function to handle variable-size data"""
        collated = {}
        
        # Handle tensor data
        tensor_keys = ['camera_images', 'camera_intrinsics', 'camera_extrinsics', 
                      'gt_boxes', 'gt_labels', 'gt_valid']
        
        for key in tensor_keys:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Handle variable-size point cloud data
        if 'lidar_points' in batch[0]:
            collated['lidar_points'] = [item['lidar_points'] for item in batch]
        if 'radar_points' in batch[0]:
            collated['radar_points'] = [item['radar_points'] for item in batch]
        
        # Handle metadata
        for key in ['sample_token', 'timestamp', 'scene_token', 'metadata']:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
        
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    # Example usage and testing
    config = NuScenesConfig(
        data_root="data/nuscenes",
        version="v1.0-trainval",
        image_size=(224, 224),
        bev_size=(48, 48),
        num_queries=100
    )
    
    try:
        dataset = NuScenesMultiModalDataset(config=config, split="train")
        dataloader = create_nuscenes_dataloader(config=config, split="train", batch_size=2)
        
        # Test loading a batch
        for batch in dataloader:
            print(f"Loaded batch with {len(batch['sample_token'])} samples")
            print(f"Camera images shape: {batch['camera_images'].shape}")
            print(f"GT boxes shape: {batch['gt_boxes'].shape}")
            break
            
    except Exception as e:
        logger.error(f"Failed to test dataset: {e}") 