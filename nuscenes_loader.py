#!/usr/bin/env python3
"""
nuScenes Data Loader for BEVNeXt-SAM2 Evaluation

This module loads real nuScenes dataset samples for evaluation instead of synthetic data.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Any

class NuScenesLoader:
    """nuScenes dataset loader for evaluation"""
    
    def __init__(self, data_root: str = "data", version: str = "v1.0-mini"):
        """
        Initialize nuScenes loader
        
        Args:
            data_root: Path to the nuScenes data directory
            version: nuScenes version (e.g., 'v1.0-mini', 'v1.0-trainval')
        """
        self.data_root = Path(data_root)
        self.version = version
        self.annotations_path = self.data_root / version
        
        # Load all annotation tables
        self.sample = self._load_table('sample.json')
        self.sample_data = self._load_table('sample_data.json')
        self.sample_annotation = self._load_table('sample_annotation.json')
        self.instance = self._load_table('instance.json')
        self.category = self._load_table('category.json')
        self.calibrated_sensor = self._load_table('calibrated_sensor.json')
        self.ego_pose = self._load_table('ego_pose.json')
        self.scene = self._load_table('scene.json')
        
        # Create lookup dictionaries for fast access
        self.sample_data_by_token = {record['token']: record for record in self.sample_data}
        self.sample_annotation_by_token = {record['token']: record for record in self.sample_annotation}
        self.calibrated_sensor_by_token = {record['token']: record for record in self.calibrated_sensor}
        self.ego_pose_by_token = {record['token']: record for record in self.ego_pose}
        self.instance_by_token = {record['token']: record for record in self.instance}
        self.category_by_token = {record['token']: record for record in self.category}
        
        # Camera names
        self.camera_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
        ]
        
        print(f"✅ nuScenes {version} dataset loaded:")
        print(f"   └─ Samples: {len(self.sample)}")
        print(f"   └─ Annotations: {len(self.sample_annotation)}")
        print(f"   └─ Categories: {len(self.category)}")
        
    def _load_table(self, filename: str) -> List[Dict]:
        """Load a nuScenes table from JSON file"""
        filepath = self.annotations_path / filename
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            return []
    
    def get_sample_data(self, sample_token: str, sensor_name: str) -> Dict:
        """Get sample data for a specific sensor"""
        # Find sample_data records that match both sample_token and sensor_name
        # Prefer 'samples' (key frames) over 'sweeps' (intermediate frames)
        samples_data = []
        sweeps_data = []
        
        for sample_data in self.sample_data:
            if sample_data.get('sample_token') == sample_token:
                # Extract sensor name from filename path
                filename = sample_data.get('filename', '')
                if filename:
                    parts = filename.split('/')
                    if len(parts) >= 2:
                        file_sensor_name = parts[1]  # e.g., 'CAM_FRONT', 'RADAR_FRONT', etc.
                        if file_sensor_name == sensor_name:
                            if parts[0] == 'samples':
                                samples_data.append(sample_data)
                            elif parts[0] == 'sweeps':
                                sweeps_data.append(sample_data)
        
        # Return key frame (samples) if available, otherwise use sweeps
        if samples_data:
            return samples_data[0]  # Use first samples match
        elif sweeps_data:
            return sweeps_data[0]   # Use first sweeps match
        return None
    
    def get_sample_annotations(self, sample_token: str) -> List[Dict]:
        """Get all 3D annotations for a sample"""
        annotations = []
        for annotation in self.sample_annotation:
            if annotation.get('sample_token') == sample_token:
                annotations.append(annotation)
        return annotations
    
    def load_camera_image(self, sample_data_record: Dict) -> np.ndarray:
        """Load camera image from sample data record"""
        if not sample_data_record:
            return None
            
        filename = sample_data_record.get('filename', '')
        image_path = self.data_root / filename
        
        try:
            # Load image using OpenCV
            img = cv2.imread(str(image_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            return img
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            return None
    
    def get_camera_intrinsic(self, sample_data_record: Dict) -> np.ndarray:
        """Get camera intrinsic matrix"""
        if not sample_data_record:
            return np.eye(3)
            
        calibrated_sensor_token = sample_data_record.get('calibrated_sensor_token')
        calibrated_sensor = self.calibrated_sensor_by_token.get(calibrated_sensor_token, {})
        
        camera_intrinsic = calibrated_sensor.get('camera_intrinsic', [[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        return np.array(camera_intrinsic)
    
    def get_camera_extrinsic(self, sample_data_record: Dict) -> np.ndarray:
        """Get camera extrinsic matrix (sensor to ego vehicle)"""
        if not sample_data_record:
            return np.eye(4)
            
        calibrated_sensor_token = sample_data_record.get('calibrated_sensor_token')
        calibrated_sensor = self.calibrated_sensor_by_token.get(calibrated_sensor_token, {})
        
        # Get ego pose for this timestamp
        ego_pose_token = sample_data_record.get('ego_pose_token')
        ego_pose = self.ego_pose_by_token.get(ego_pose_token, {})
        
        # Get sensor to ego transformation
        sensor_rotation = calibrated_sensor.get('rotation', [1, 0, 0, 0])  # quaternion [w, x, y, z]
        sensor_translation = calibrated_sensor.get('translation', [0, 0, 0])
        
        # Get ego to global transformation
        ego_rotation = ego_pose.get('rotation', [1, 0, 0, 0])
        ego_translation = ego_pose.get('translation', [0, 0, 0])
        
        # Convert quaternions to rotation matrices
        def quat_to_rotation_matrix(q):
            w, x, y, z = q
            return np.array([
                [1-2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1-2*(x**2 + z**2), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x**2 + y**2)]
            ])
        
        # Sensor to ego transformation
        sensor_to_ego_rot = quat_to_rotation_matrix(sensor_rotation)
        sensor_to_ego = np.eye(4)
        sensor_to_ego[:3, :3] = sensor_to_ego_rot
        sensor_to_ego[:3, 3] = sensor_translation
        
        # Ego to global transformation  
        ego_to_global_rot = quat_to_rotation_matrix(ego_rotation)
        ego_to_global = np.eye(4)
        ego_to_global[:3, :3] = ego_to_global_rot
        ego_to_global[:3, 3] = ego_translation
        
        # Combined transformation: sensor to global
        sensor_to_global = np.dot(ego_to_global, sensor_to_ego)
        
        return sensor_to_global
    
    def annotation_to_bbox3d(self, annotation: Dict) -> np.ndarray:
        """Convert nuScenes annotation to 3D bounding box format [x, y, z, w, l, h, yaw]"""
        translation = annotation.get('translation', [0, 0, 0])  # [x, y, z] in global coordinates
        size = annotation.get('size', [1, 1, 1])  # [width, length, height]
        rotation = annotation.get('rotation', [1, 0, 0, 0])  # quaternion [w, x, y, z]
        
        # Convert quaternion to yaw angle (rotation around z-axis)
        w, x, y, z = rotation
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        
        # nuScenes format: [center_x, center_y, center_z, width, length, height, yaw]
        bbox_3d = [
            translation[0],  # x
            translation[1],  # y  
            translation[2],  # z
            size[0],         # width
            size[1],         # length
            size[2],         # height
            yaw              # rotation around z-axis
        ]
        
        return np.array(bbox_3d)
    
    def get_category_name(self, annotation: Dict) -> str:
        """Get category name from annotation"""
        category_token = annotation.get('category_token')
        category = self.category_by_token.get(category_token, {})
        return category.get('name', 'unknown')
    
    def create_evaluation_sample(self, sample_idx: int) -> Dict:
        """Create a single evaluation sample from nuScenes data"""
        if sample_idx >= len(self.sample):
            return None
            
        sample = self.sample[sample_idx]
        sample_token = sample['token']
        
        # Load camera images
        camera_images = []
        camera_intrinsics = []
        camera_extrinsics = []
        
        for cam_name in self.camera_names:
            sample_data_record = self.get_sample_data(sample_token, cam_name)
            
            # Load image
            img = self.load_camera_image(sample_data_record)
            if img is not None:
                camera_images.append(img)
            else:
                # Create placeholder if image loading fails
                camera_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Get camera calibration
            intrinsic = self.get_camera_intrinsic(sample_data_record)
            extrinsic = self.get_camera_extrinsic(sample_data_record)
            
            camera_intrinsics.append(intrinsic)
            camera_extrinsics.append(extrinsic)
        
        # Load annotations (3D bounding boxes)
        annotations = self.get_sample_annotations(sample_token)
        gt_boxes = []
        gt_labels = []
        gt_category_names = []
        
        for annotation in annotations:
            bbox_3d = self.annotation_to_bbox3d(annotation)
            category_name = self.get_category_name(annotation)
            
            gt_boxes.append(bbox_3d)
            
            # Map category to label index (simplified mapping)
            category_mapping = {
                'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3,
                'construction_vehicle': 4, 'pedestrian': 5, 'motorcycle': 6, 'bicycle': 7,
                'traffic_cone': 8, 'barrier': 9
            }
            
            label = category_mapping.get(category_name.split('.')[0], 0)  # Default to car
            gt_labels.append(label)
            gt_category_names.append(category_name)
        
        # Create sample dictionary
        evaluation_sample = {
            'sample_id': f'nuscenes_{sample_idx:04d}',
            'sample_token': sample_token,
            'camera_images': camera_images,
            'gt_boxes': torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.empty(0, 7),
            'gt_labels': torch.tensor(gt_labels, dtype=torch.long) if gt_labels else torch.empty(0, dtype=torch.long),
            'gt_category_names': gt_category_names,
            'camera_intrinsics': torch.tensor(np.stack(camera_intrinsics), dtype=torch.float32),
            'camera_extrinsics': torch.tensor(np.stack(camera_extrinsics), dtype=torch.float32),
            'timestamp': sample.get('timestamp', 0),
        }
        
        return evaluation_sample
    
    def create_evaluation_dataset(self, num_samples: int = 50) -> List[Dict]:
        """Create evaluation dataset from nuScenes samples"""
        print(f"🔄 Loading nuScenes dataset samples ({num_samples} samples)...")
        
        dataset = []
        max_samples = min(num_samples, len(self.sample))
        
        for i in range(max_samples):
            try:
                sample = self.create_evaluation_sample(i)
                if sample is not None:
                    dataset.append(sample)
                    
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   └─ Loaded {i + 1}/{max_samples} samples...")
                    
            except Exception as e:
                print(f"Warning: Could not load sample {i}: {e}")
                continue
        
        print(f"✅ nuScenes dataset loaded with {len(dataset)} samples")
        return dataset

def load_nuscenes_test_dataset(data_root: str = "data", num_samples: int = 50) -> List[Dict]:
    """
    Load nuScenes test dataset for evaluation
    
    Args:
        data_root: Path to nuScenes data directory
        num_samples: Number of samples to load
        
    Returns:
        List of evaluation samples with real nuScenes data
    """
    loader = NuScenesLoader(data_root)
    return loader.create_evaluation_dataset(num_samples) 