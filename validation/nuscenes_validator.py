#!/usr/bin/env python3
"""
nuScenes Dataset Validation and Integrity Checker

This script provides comprehensive validation for nuScenes v1.0 dataset including:
- Token linkage integrity validation
- Sensor data file existence verification
- Annotation-sensor data correspondence checks
- Data quality assessment and reporting

Author: Senior Python Programmer & Autonomous Driving Dataset Expert
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.splits import create_splits_scenes
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Data class to store validation results"""
    test_name: str
    passed: bool
    error_count: int
    warning_count: int
    details: Dict[str, Any]
    timestamp: str
    
@dataclass
class DatasetHealthReport:
    """Comprehensive dataset health report"""
    total_scenes: int
    total_samples: int
    total_annotations: int
    sensor_data_counts: Dict[str, int]
    missing_files: List[str]
    broken_token_chains: List[str]
    annotation_issues: List[str]
    quality_scores: Dict[str, float]
    recommendations: List[str]


class NuScenesTokenValidator:
    """Validates token-based associations in nuScenes dataset"""
    
    def __init__(self, nusc: NuScenes, verbose: bool = True):
        self.nusc = nusc
        self.verbose = verbose
        self.validation_results = []
        
    def validate_scene_sample_chains(self) -> ValidationResult:
        """Validate that scene->sample token chains are intact"""
        broken_chains = []
        total_scenes = 0
        
        for scene in tqdm(self.nusc.scene, desc="Validating scene-sample chains", disable=not self.verbose):
            total_scenes += 1
            scene_token = scene['token']
            sample_token = scene['first_sample_token']
            sample_count = 0
            
            # Traverse the sample chain
            while sample_token != '':
                try:
                    sample = self.nusc.get('sample', sample_token)
                    sample_count += 1
                    sample_token = sample['next']
                    
                    # Check if we exceed expected number of samples
                    if sample_count > scene['nbr_samples']:
                        broken_chains.append({
                            'scene_token': scene_token,
                            'scene_name': scene['name'],
                            'issue': 'Sample count exceeds expected',
                            'expected': scene['nbr_samples'],
                            'actual': sample_count
                        })
                        break
                        
                except Exception as e:
                    broken_chains.append({
                        'scene_token': scene_token,
                        'scene_name': scene['name'],
                        'issue': f'Broken token chain at sample {sample_count}',
                        'error': str(e)
                    })
                    break
            
            # Check if sample count matches expected
            if sample_count != scene['nbr_samples'] and sample_count <= scene['nbr_samples']:
                broken_chains.append({
                    'scene_token': scene_token,
                    'scene_name': scene['name'],
                    'issue': 'Sample count mismatch',
                    'expected': scene['nbr_samples'],
                    'actual': sample_count
                })
        
        result = ValidationResult(
            test_name="Scene-Sample Chain Validation",
            passed=len(broken_chains) == 0,
            error_count=len(broken_chains),
            warning_count=0,
            details={'broken_chains': broken_chains, 'total_scenes': total_scenes},
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_sample_data_associations(self) -> ValidationResult:
        """Validate sample to sample_data associations"""
        missing_associations = []
        sensor_channels = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                          'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP']
        
        for sample in tqdm(self.nusc.sample, desc="Validating sample-data associations", disable=not self.verbose):
            sample_token = sample['token']
            
            for sensor_channel in sensor_channels:
                if sensor_channel not in sample['data']:
                    missing_associations.append({
                        'sample_token': sample_token,
                        'missing_sensor': sensor_channel,
                        'timestamp': sample['timestamp']
                    })
                else:
                    # Verify that the sample_data record exists and is valid
                    sample_data_token = sample['data'][sensor_channel]
                    try:
                        sample_data = self.nusc.get('sample_data', sample_data_token)
                        
                        # Check if it's a key frame and points back to the sample
                        if not sample_data['is_key_frame']:
                            missing_associations.append({
                                'sample_token': sample_token,
                                'sensor': sensor_channel,
                                'issue': 'Sample data is not marked as key frame',
                                'sample_data_token': sample_data_token
                            })
                        
                        if sample_data['sample_token'] != sample_token:
                            missing_associations.append({
                                'sample_token': sample_token,
                                'sensor': sensor_channel,
                                'issue': 'Sample data does not point back to correct sample',
                                'expected_sample': sample_token,
                                'actual_sample': sample_data['sample_token']
                            })
                            
                    except Exception as e:
                        missing_associations.append({
                            'sample_token': sample_token,
                            'sensor': sensor_channel,
                            'issue': f'Invalid sample_data token: {str(e)}',
                            'sample_data_token': sample_data_token
                        })
        
        result = ValidationResult(
            test_name="Sample-Data Association Validation",
            passed=len(missing_associations) == 0,
            error_count=len(missing_associations),
            warning_count=0,
            details={'missing_associations': missing_associations},
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_annotation_sample_associations(self) -> ValidationResult:
        """Validate sample_annotation to sample associations"""
        orphaned_annotations = []
        
        for annotation in tqdm(self.nusc.sample_annotation, desc="Validating annotation-sample associations", disable=not self.verbose):
            sample_token = annotation['sample_token']
            
            try:
                sample = self.nusc.get('sample', sample_token)
                
                # Check if the sample includes this annotation
                if annotation['token'] not in sample['anns']:
                    orphaned_annotations.append({
                        'annotation_token': annotation['token'],
                        'sample_token': sample_token,
                        'category': annotation['category_name'],
                        'issue': 'Annotation not listed in sample.anns'
                    })
                    
            except Exception as e:
                orphaned_annotations.append({
                    'annotation_token': annotation['token'],
                    'sample_token': sample_token,
                    'category': annotation['category_name'],
                    'issue': f'Invalid sample token: {str(e)}'
                })
        
        result = ValidationResult(
            test_name="Annotation-Sample Association Validation",
            passed=len(orphaned_annotations) == 0,
            error_count=len(orphaned_annotations),
            warning_count=0,
            details={'orphaned_annotations': orphaned_annotations},
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        self.validation_results.append(result)
        return result


class NuScenesFileValidator:
    """Validates file existence and format correctness for nuScenes dataset"""
    
    def __init__(self, nusc: NuScenes, verbose: bool = True):
        self.nusc = nusc
        self.verbose = verbose
        self.validation_results = []
        
    def validate_sensor_file_existence(self) -> ValidationResult:
        """Check that all sensor data files exist and are accessible"""
        missing_files = []
        corrupt_files = []
        file_stats = defaultdict(int)
        
        for sample_data in tqdm(self.nusc.sample_data, desc="Validating sensor files", disable=not self.verbose):
            if sample_data['is_key_frame']:  # Focus on key frames
                file_path = Path(self.nusc.dataroot) / sample_data['filename']
                sensor_token = sample_data['sensor_token']
                sensor = self.nusc.get('sensor', sensor_token)
                sensor_channel = sensor['channel']
                
                file_stats[sensor_channel] += 1
                
                # Check file existence
                if not file_path.exists():
                    missing_files.append({
                        'filename': sample_data['filename'],
                        'sensor': sensor_channel,
                        'sample_data_token': sample_data['token'],
                        'timestamp': sample_data['timestamp']
                    })
                    continue
                
                # Check file size and basic format
                try:
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        corrupt_files.append({
                            'filename': sample_data['filename'],
                            'sensor': sensor_channel,
                            'issue': 'Zero file size'
                        })
                    elif file_size < 1024:  # Suspiciously small
                        corrupt_files.append({
                            'filename': sample_data['filename'],
                            'sensor': sensor_channel,
                            'issue': f'Suspiciously small file: {file_size} bytes'
                        })
                        
                    # Basic format validation
                    if sensor_channel.startswith('CAM'):
                        # Validate image file
                        try:
                            from PIL import Image
                            with Image.open(file_path) as img:
                                if img.size != (1600, 900):
                                    corrupt_files.append({
                                        'filename': sample_data['filename'],
                                        'sensor': sensor_channel,
                                        'issue': f'Unexpected image size: {img.size}, expected (1600, 900)'
                                    })
                        except Exception as e:
                            corrupt_files.append({
                                'filename': sample_data['filename'],
                                'sensor': sensor_channel,
                                'issue': f'Cannot open as image: {str(e)}'
                            })
                    
                    elif sensor_channel == 'LIDAR_TOP':
                        # Validate LiDAR file
                        try:
                            pc = LidarPointCloud.from_file(str(file_path))
                            if pc.points.shape[1] < 1000:  # Suspiciously few points
                                corrupt_files.append({
                                    'filename': sample_data['filename'],
                                    'sensor': sensor_channel,
                                    'issue': f'Very few LiDAR points: {pc.points.shape[1]}'
                                })
                        except Exception as e:
                            corrupt_files.append({
                                'filename': sample_data['filename'],
                                'sensor': sensor_channel,
                                'issue': f'Cannot load LiDAR data: {str(e)}'
                            })
                    
                    elif sensor_channel.startswith('RADAR'):
                        # Validate radar file
                        try:
                            pc = RadarPointCloud.from_file(str(file_path))
                            if pc.points.shape[1] == 0:
                                corrupt_files.append({
                                    'filename': sample_data['filename'],
                                    'sensor': sensor_channel,
                                    'issue': 'No radar points found'
                                })
                        except Exception as e:
                            corrupt_files.append({
                                'filename': sample_data['filename'],
                                'sensor': sensor_channel,
                                'issue': f'Cannot load radar data: {str(e)}'
                            })
                            
                except Exception as e:
                    corrupt_files.append({
                        'filename': sample_data['filename'],
                        'sensor': sensor_channel,
                        'issue': f'File access error: {str(e)}'
                    })
        
        total_issues = len(missing_files) + len(corrupt_files)
        
        result = ValidationResult(
            test_name="Sensor File Validation",
            passed=total_issues == 0,
            error_count=len(missing_files),
            warning_count=len(corrupt_files),
            details={
                'missing_files': missing_files,
                'corrupt_files': corrupt_files,
                'file_stats': dict(file_stats)
            },
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_annotation_geometry(self) -> ValidationResult:
        """Validate 3D bounding box annotation geometry"""
        invalid_annotations = []
        
        for annotation in tqdm(self.nusc.sample_annotation, desc="Validating annotation geometry", disable=not self.verbose):
            issues = []
            
            # Check size validity
            size = annotation['size']
            if any(s <= 0 for s in size):
                issues.append('Non-positive size values')
            if any(s > 50 for s in size):  # Unreasonably large objects
                issues.append(f'Unreasonably large size: {size}')
            
            # Check translation validity
            translation = annotation['translation']
            if abs(translation[0]) > 1000 or abs(translation[1]) > 1000:
                issues.append(f'Object very far from ego vehicle: {translation[:2]}')
            if translation[2] < -10 or translation[2] > 10:
                issues.append(f'Unusual height: {translation[2]}')
            
            # Check rotation validity
            rotation = annotation['rotation']
            if len(rotation) != 4:
                issues.append('Invalid quaternion format')
            else:
                # Check if quaternion is normalized
                norm = sum(x*x for x in rotation) ** 0.5
                if abs(norm - 1.0) > 0.1:
                    issues.append(f'Quaternion not normalized: norm={norm}')
            
            # Check velocity if present
            if 'velocity' in annotation:
                velocity = annotation['velocity']
                if len(velocity) >= 2:
                    speed = (velocity[0]**2 + velocity[1]**2)**0.5
                    if speed > 50:  # > 180 km/h
                        issues.append(f'Unrealistic speed: {speed:.1f} m/s')
            
            if issues:
                invalid_annotations.append({
                    'annotation_token': annotation['token'],
                    'category': annotation['category_name'],
                    'sample_token': annotation['sample_token'],
                    'issues': issues
                })
        
        result = ValidationResult(
            test_name="Annotation Geometry Validation",
            passed=len(invalid_annotations) == 0,
            error_count=0,
            warning_count=len(invalid_annotations),
            details={'invalid_annotations': invalid_annotations},
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        self.validation_results.append(result)
        return result


class NuScenesDataQualityAnalyzer:
    """Analyzes data quality and generates comprehensive reports"""
    
    def __init__(self, nusc: NuScenes, verbose: bool = True):
        self.nusc = nusc
        self.verbose = verbose
        
    def analyze_dataset_completeness(self) -> Dict[str, Any]:
        """Analyze dataset completeness and coverage"""
        stats = {
            'scene_coverage': self._analyze_scene_coverage(),
            'sensor_coverage': self._analyze_sensor_coverage(),
            'annotation_coverage': self._analyze_annotation_coverage(),
            'temporal_coverage': self._analyze_temporal_coverage()
        }
        return stats
    
    def _analyze_scene_coverage(self) -> Dict[str, Any]:
        """Analyze scene distribution and coverage"""
        location_counts = defaultdict(int)
        time_of_day_counts = defaultdict(int)
        weather_counts = defaultdict(int)
        scene_lengths = []
        
        for scene in self.nusc.scene:
            scene_lengths.append(scene['nbr_samples'])
            log = self.nusc.get('log', scene['log_token'])
            location_counts[log['location']] += 1
            
            # Analyze description for time of day and weather
            description = scene['description'].lower()
            if any(word in description for word in ['night', 'dark', 'evening']):
                time_of_day_counts['night'] += 1
            elif any(word in description for word in ['day', 'afternoon', 'morning']):
                time_of_day_counts['day'] += 1
            else:
                time_of_day_counts['unknown'] += 1
                
            if any(word in description for word in ['rain', 'wet', 'storm']):
                weather_counts['rain'] += 1
            elif any(word in description for word in ['sun', 'clear', 'bright']):
                weather_counts['clear'] += 1
            else:
                weather_counts['unknown'] += 1
        
        return {
            'total_scenes': len(self.nusc.scene),
            'locations': dict(location_counts),
            'time_of_day': dict(time_of_day_counts),
            'weather': dict(weather_counts),
            'scene_length_stats': {
                'mean': np.mean(scene_lengths),
                'std': np.std(scene_lengths),
                'min': np.min(scene_lengths),
                'max': np.max(scene_lengths)
            }
        }
    
    def _analyze_sensor_coverage(self) -> Dict[str, Any]:
        """Analyze sensor data coverage and quality"""
        sensor_stats = defaultdict(lambda: {'count': 0, 'sizes': []})
        
        for sample_data in self.nusc.sample_data:
            if sample_data['is_key_frame']:
                sensor = self.nusc.get('sensor', sample_data['sensor_token'])
                channel = sensor['channel']
                sensor_stats[channel]['count'] += 1
                
                # Check file size if file exists
                file_path = Path(self.nusc.dataroot) / sample_data['filename']
                if file_path.exists():
                    sensor_stats[channel]['sizes'].append(file_path.stat().st_size)
        
        # Calculate statistics
        for channel in sensor_stats:
            sizes = sensor_stats[channel]['sizes']
            if sizes:
                sensor_stats[channel]['size_stats'] = {
                    'mean_mb': np.mean(sizes) / (1024*1024),
                    'std_mb': np.std(sizes) / (1024*1024),
                    'min_mb': np.min(sizes) / (1024*1024),
                    'max_mb': np.max(sizes) / (1024*1024)
                }
            del sensor_stats[channel]['sizes']  # Remove raw data
        
        return dict(sensor_stats)
    
    def _analyze_annotation_coverage(self) -> Dict[str, Any]:
        """Analyze annotation coverage and distribution"""
        category_counts = defaultdict(int)
        size_distributions = defaultdict(list)
        samples_with_annotations = set()
        
        for annotation in self.nusc.sample_annotation:
            category = annotation['category_name']
            category_counts[category] += 1
            
            # Collect size information
            size = annotation['size']
            volume = size[0] * size[1] * size[2]
            size_distributions[category].append(volume)
            
            samples_with_annotations.add(annotation['sample_token'])
        
        # Calculate size statistics per category
        size_stats = {}
        for category, volumes in size_distributions.items():
            size_stats[category] = {
                'mean_volume': np.mean(volumes),
                'std_volume': np.std(volumes),
                'count': len(volumes)
            }
        
        annotation_coverage_ratio = len(samples_with_annotations) / len(self.nusc.sample)
        
        return {
            'category_counts': dict(category_counts),
            'size_statistics': size_stats,
            'annotation_coverage_ratio': annotation_coverage_ratio,
            'samples_with_annotations': len(samples_with_annotations),
            'total_samples': len(self.nusc.sample)
        }
    
    def _analyze_temporal_coverage(self) -> Dict[str, Any]:
        """Analyze temporal distribution of data"""
        timestamps = [sample['timestamp'] for sample in self.nusc.sample]
        timestamps.sort()
        
        # Calculate intervals
        intervals = np.diff(timestamps) / 1e6  # Convert to seconds
        
        # Detect gaps
        large_gaps = []
        gap_threshold = 60  # seconds
        for i, interval in enumerate(intervals):
            if interval > gap_threshold:
                large_gaps.append({
                    'gap_duration_sec': interval,
                    'before_timestamp': timestamps[i],
                    'after_timestamp': timestamps[i+1]
                })
        
        return {
            'total_duration_hours': (timestamps[-1] - timestamps[0]) / 1e6 / 3600,
            'sample_interval_stats': {
                'mean_sec': np.mean(intervals),
                'std_sec': np.std(intervals),
                'min_sec': np.min(intervals),
                'max_sec': np.max(intervals)
            },
            'large_gaps': large_gaps,
            'total_samples': len(timestamps)
        }
    
    def generate_quality_scores(self, validation_results: List[ValidationResult]) -> Dict[str, float]:
        """Generate overall quality scores based on validation results"""
        scores = {}
        
        # Completeness score (0-100)
        total_errors = sum(result.error_count for result in validation_results)
        total_warnings = sum(result.warning_count for result in validation_results)
        total_tests = len(validation_results)
        
        if total_tests > 0:
            passed_tests = sum(1 for result in validation_results if result.passed)
            scores['completeness'] = (passed_tests / total_tests) * 100
        else:
            scores['completeness'] = 0
        
        # Data integrity score (penalize errors more than warnings)
        integrity_penalty = total_errors * 2 + total_warnings
        max_possible_penalty = len(self.nusc.sample) * 0.1  # Assume max 10% issues
        scores['integrity'] = max(0, 100 - (integrity_penalty / max_possible_penalty) * 100)
        
        # Coverage score based on sensor and annotation coverage
        sensor_coverage = self._calculate_sensor_coverage_score()
        annotation_coverage = self._calculate_annotation_coverage_score()
        scores['coverage'] = (sensor_coverage + annotation_coverage) / 2
        
        # Overall score
        scores['overall'] = (scores['completeness'] + scores['integrity'] + scores['coverage']) / 3
        
        return scores
    
    def _calculate_sensor_coverage_score(self) -> float:
        """Calculate sensor coverage score"""
        expected_sensors = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                           'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP']
        
        sensor_coverage = {}
        for sample in self.nusc.sample:
            for sensor in expected_sensors:
                if sensor not in sensor_coverage:
                    sensor_coverage[sensor] = 0
                if sensor in sample['data']:
                    sensor_coverage[sensor] += 1
        
        total_samples = len(self.nusc.sample)
        coverage_ratios = [count / total_samples for count in sensor_coverage.values()]
        return np.mean(coverage_ratios) * 100
    
    def _calculate_annotation_coverage_score(self) -> float:
        """Calculate annotation coverage score"""
        samples_with_annotations = set()
        for annotation in self.nusc.sample_annotation:
            samples_with_annotations.add(annotation['sample_token'])
        
        coverage_ratio = len(samples_with_annotations) / len(self.nusc.sample)
        return coverage_ratio * 100


class NuScenesValidationReport:
    """Generates comprehensive validation reports"""
    
    def __init__(self, validation_results: List[ValidationResult], 
                 quality_analysis: Dict[str, Any],
                 quality_scores: Dict[str, float]):
        self.validation_results = validation_results
        self.quality_analysis = quality_analysis
        self.quality_scores = quality_scores
        
    def generate_text_report(self) -> str:
        """Generate a comprehensive text report"""
        report = []
        report.append("=" * 80)
        report.append("NUSCENES DATASET VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Quality Scores Summary
        report.append("QUALITY SCORES")
        report.append("-" * 40)
        for score_name, score_value in self.quality_scores.items():
            status = "EXCELLENT" if score_value >= 90 else "GOOD" if score_value >= 70 else "FAIR" if score_value >= 50 else "POOR"
            report.append(f"{score_name.title():15}: {score_value:6.1f}% ({status})")
        report.append("")
        
        # Validation Results Summary
        report.append("VALIDATION RESULTS")
        report.append("-" * 40)
        total_errors = sum(result.error_count for result in self.validation_results)
        total_warnings = sum(result.warning_count for result in self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result.passed)
        
        report.append(f"Tests Run:    {len(self.validation_results)}")
        report.append(f"Tests Passed: {passed_tests}")
        report.append(f"Total Errors: {total_errors}")
        report.append(f"Total Warnings: {total_warnings}")
        report.append("")
        
        # Detailed Results
        for result in self.validation_results:
            status = "PASS" if result.passed else "FAIL"
            report.append(f"{result.test_name}: {status}")
            if result.error_count > 0:
                report.append(f"  Errors: {result.error_count}")
            if result.warning_count > 0:
                report.append(f"  Warnings: {result.warning_count}")
        report.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        # Check for critical issues
        for result in self.validation_results:
            if result.error_count > 0:
                if "File Validation" in result.test_name:
                    recommendations.append("Re-download or restore missing/corrupted sensor data files")
                elif "Chain Validation" in result.test_name:
                    recommendations.append("Verify dataset extraction process - some token chains may be broken")
                elif "Association Validation" in result.test_name:
                    recommendations.append("Check dataset integrity - annotation-sample associations may be corrupted")
        
        # Check quality scores
        if self.quality_scores['coverage'] < 70:
            recommendations.append("Dataset has low sensor/annotation coverage - consider using full dataset")
        
        if self.quality_scores['integrity'] < 80:
            recommendations.append("Dataset integrity issues detected - consider re-extracting dataset")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Dataset validation passed - ready for training")
        
        return recommendations
    
    def save_detailed_report(self, output_path: str):
        """Save detailed validation report with data"""
        report_data = {
            'validation_results': [asdict(result) for result in self.validation_results],
            'quality_analysis': self.quality_analysis,
            'quality_scores': self.quality_scores,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Detailed validation report saved to: {output_path}")


def run_complete_validation(data_root: str, 
                          version: str = "v1.0-trainval",
                          output_dir: str = "validation_reports",
                          verbose: bool = True) -> NuScenesValidationReport:
    """
    Run complete nuScenes dataset validation
    
    Args:
        data_root: Path to nuScenes dataset
        version: Dataset version
        output_dir: Directory to save reports
        verbose: Enable verbose output
        
    Returns:
        NuScenesValidationReport object
    """
    logger.info(f"Starting complete validation of nuScenes {version} at {data_root}")
    
    # Initialize nuScenes
    try:
        nusc = NuScenes(version=version, dataroot=data_root, verbose=verbose)
    except Exception as e:
        logger.error(f"Failed to load nuScenes dataset: {e}")
        raise
    
    # Run token validation
    logger.info("Running token validation tests...")
    token_validator = NuScenesTokenValidator(nusc, verbose)
    token_results = [
        token_validator.validate_scene_sample_chains(),
        token_validator.validate_sample_data_associations(),
        token_validator.validate_annotation_sample_associations()
    ]
    
    # Run file validation
    logger.info("Running file validation tests...")
    file_validator = NuScenesFileValidator(nusc, verbose)
    file_results = [
        file_validator.validate_sensor_file_existence(),
        file_validator.validate_annotation_geometry()
    ]
    
    # Combine all results
    all_results = token_results + file_results
    
    # Run quality analysis
    logger.info("Running quality analysis...")
    quality_analyzer = NuScenesDataQualityAnalyzer(nusc, verbose)
    quality_analysis = quality_analyzer.analyze_dataset_completeness()
    quality_scores = quality_analyzer.generate_quality_scores(all_results)
    
    # Generate report
    report = NuScenesValidationReport(all_results, quality_analysis, quality_scores)
    
    # Save reports
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text report
    text_report = report.generate_text_report()
    text_path = os.path.join(output_dir, f"validation_report_{version}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(text_path, 'w') as f:
        f.write(text_report)
    logger.info(f"Text report saved to: {text_path}")
    
    # Save detailed JSON report
    json_path = os.path.join(output_dir, f"validation_detailed_{version}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    report.save_detailed_report(json_path)
    
    # Print summary
    print(text_report)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='nuScenes Dataset Validation')
    parser.add_argument('--data-root', default='data/nuscenes', help='Path to nuScenes dataset')
    parser.add_argument('--version', default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--output-dir', default='validation_reports', help='Output directory for reports')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        report = run_complete_validation(
            data_root=args.data_root,
            version=args.version,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Quick summary
        overall_score = report.quality_scores['overall']
        if overall_score >= 90:
            logger.info("Dataset validation PASSED - Excellent quality")
        elif overall_score >= 70:
            logger.info("Dataset validation PASSED - Good quality")
        elif overall_score >= 50:
            logger.info("Dataset validation PASSED with warnings - Fair quality")
        else:
            logger.error("Dataset validation FAILED - Poor quality")
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise 