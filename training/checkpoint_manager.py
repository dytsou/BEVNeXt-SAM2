#!/usr/bin/env python3
"""
Enhanced Checkpoint Manager for Robust Training Resume

This module provides comprehensive checkpoint management with validation,
automatic cleanup, metadata tracking, and robust error handling.

Author: Senior Python Programmer & AI Training Expert
"""

import os
import json
import time
import hashlib
import logging
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import torch
import pickle
import gzip
import shutil
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint tracking"""
    checkpoint_path: str
    epoch: int
    step: Optional[int]
    timestamp: str
    model_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    optimizer_type: str
    scheduler_type: str
    total_params: int
    model_size_mb: float
    validation_loss: Optional[float]
    is_best: bool
    resume_count: int
    creation_time: float
    checksum: str
    compressed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(**data)


class CheckpointManager:
    """Enhanced checkpoint manager with validation and auto-cleanup"""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_latest: int = 5,
        keep_best: int = 3,
        keep_epoch_interval: int = 10,
        enable_compression: bool = False,
        async_save: bool = True,
        max_disk_usage_gb: float = 50.0
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            keep_latest: Number of latest checkpoints to keep
            keep_best: Number of best checkpoints to keep
            keep_epoch_interval: Save every N epochs permanently
            enable_compression: Use gzip compression for checkpoints
            async_save: Enable asynchronous checkpoint saving
            max_disk_usage_gb: Maximum disk usage in GB
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_latest = keep_latest
        self.keep_best = keep_best
        self.keep_epoch_interval = keep_epoch_interval
        self.enable_compression = enable_compression
        self.async_save = async_save
        self.max_disk_usage_gb = max_disk_usage_gb
        
        self.metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'
        self.lock_file = self.checkpoint_dir / '.checkpoint_lock'
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2) if async_save else None
        self._active_saves = set()
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
        logger.info(f"Settings: keep_latest={keep_latest}, keep_best={keep_best}, "
                   f"compression={enable_compression}, async={async_save}")

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        scheduler_state: Dict[str, Any],
        epoch: int,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        is_emergency: bool = False,
        blocking: bool = False
    ) -> Optional[Path]:
        """
        Save checkpoint with comprehensive metadata
        
        Args:
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            epoch: Current epoch
            step: Current step within epoch
            metrics: Training metrics
            model_config: Model configuration
            is_best: Whether this is the best checkpoint
            is_emergency: Whether this is an emergency save
            blocking: Force synchronous save even if async is enabled
            
        Returns:
            Path to saved checkpoint (None if async)
        """
        metrics = metrics or {}
        model_config = model_config or {}
        
        # Create checkpoint data
        checkpoint_data = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'model_config': model_config,
            'timestamp': datetime.now().isoformat(),
            'resume_count': 0,
            'is_emergency': is_emergency
        }
        
        # Generate checkpoint filename
        if is_emergency:
            filename = f'emergency_checkpoint_{int(time.time())}.pth'
        elif is_best:
            filename = 'checkpoint_best.pth'
        elif step is not None:
            filename = f'checkpoint_epoch_{epoch}_step_{step}.pth'
        else:
            filename = f'checkpoint_epoch_{epoch}.pth'
            
        if self.enable_compression:
            filename += '.gz'
            
        checkpoint_path = self.checkpoint_dir / filename
        
        # Create metadata
        metadata = self._create_metadata(
            checkpoint_path, checkpoint_data, is_best
        )
        
        if self.async_save and not blocking and not is_emergency:
            # Asynchronous save
            future = self._executor.submit(
                self._save_checkpoint_sync, checkpoint_path, checkpoint_data, metadata
            )
            self._active_saves.add(future)
            
            # Clean up completed futures
            self._cleanup_completed_saves()
            
            logger.info(f"Async checkpoint save started: {checkpoint_path}")
            return None
        else:
            # Synchronous save
            return self._save_checkpoint_sync(checkpoint_path, checkpoint_data, metadata)

    def _save_checkpoint_sync(
        self, 
        checkpoint_path: Path, 
        checkpoint_data: Dict[str, Any],
        metadata: CheckpointMetadata
    ) -> Path:
        """Synchronously save checkpoint with validation"""
        start_time = time.time()
        
        try:
            # Create temporary file for atomic write
            temp_path = checkpoint_path.with_suffix('.tmp')
            
            # Save checkpoint
            if self.enable_compression:
                with gzip.open(temp_path, 'wb') as f:
                    torch.save(checkpoint_data, f)
            else:
                torch.save(checkpoint_data, temp_path)
            
            # Validate saved checkpoint
            if not self._validate_checkpoint_file(temp_path):
                temp_path.unlink()
                raise ValueError("Checkpoint validation failed after save")
            
            # Calculate checksum
            metadata.checksum = self._calculate_checksum(temp_path)
            
            # Atomic move to final location
            temp_path.rename(checkpoint_path)
            
            # Update metadata
            self._update_metadata(metadata)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            save_time = time.time() - start_time
            logger.info(f"Checkpoint saved: {checkpoint_path} ({save_time:.2f}s)")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        auto_select: bool = True,
        validate: bool = True
    ) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """
        Load checkpoint with validation
        
        Args:
            checkpoint_path: Specific checkpoint to load (None for auto-select)
            auto_select: Automatically select best available checkpoint
            validate: Validate checkpoint before loading
            
        Returns:
            Tuple of (checkpoint_data, metadata)
        """
        if checkpoint_path is None and auto_select:
            checkpoint_path = self.find_latest_valid_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No valid checkpoints found")
        
        if checkpoint_path is None:
            raise ValueError("No checkpoint specified and auto-select failed")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Validate checkpoint if requested
        if validate and not self.validate_checkpoint(checkpoint_path):
            raise ValueError(f"Checkpoint validation failed: {checkpoint_path}")
        
        try:
            start_time = time.time()
            
            # Load checkpoint data
            if checkpoint_path.suffix == '.gz':
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint_data = torch.load(f, map_location='cpu')
            else:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Get metadata
            metadata = self._get_checkpoint_metadata(checkpoint_path)
            
            # Update resume count
            if metadata:
                metadata.resume_count += 1
                self._update_metadata(metadata)
            
            load_time = time.time() - start_time
            logger.info(f"Checkpoint loaded: {checkpoint_path} ({load_time:.2f}s)")
            
            return checkpoint_data, metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise

    def validate_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Validate checkpoint file integrity
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if checkpoint is valid
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}")
            return False
        
        try:
            # Check file size
            if checkpoint_path.stat().st_size == 0:
                logger.warning(f"Checkpoint file is empty: {checkpoint_path}")
                return False
            
            # Verify checksum if available
            metadata = self._get_checkpoint_metadata(checkpoint_path)
            if metadata and metadata.checksum:
                current_checksum = self._calculate_checksum(checkpoint_path)
                if current_checksum != metadata.checksum:
                    logger.warning(f"Checkpoint checksum mismatch: {checkpoint_path}")
                    return False
            
            # Try to load checkpoint
            return self._validate_checkpoint_file(checkpoint_path)
            
        except Exception as e:
            logger.warning(f"Checkpoint validation failed {checkpoint_path}: {e}")
            return False

    def _validate_checkpoint_file(self, checkpoint_path: Path) -> bool:
        """Validate checkpoint file by attempting to load it"""
        try:
            if checkpoint_path.suffix == '.gz':
                with gzip.open(checkpoint_path, 'rb') as f:
                    data = torch.load(f, map_location='cpu')
            else:
                data = torch.load(checkpoint_path, map_location='cpu')
            
            # Check required keys
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            for key in required_keys:
                if key not in data:
                    logger.warning(f"Missing required key '{key}' in checkpoint")
                    return False
            
            # Verify state dicts are valid
            if not isinstance(data['model_state_dict'], dict):
                logger.warning("Invalid model_state_dict format")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Checkpoint file validation failed: {e}")
            return False

    def find_latest_valid_checkpoint(self) -> Optional[Path]:
        """Find the most recent valid checkpoint"""
        valid_checkpoints = []
        
        for metadata in self.metadata.values():
            checkpoint_path = Path(metadata.checkpoint_path)
            if self.validate_checkpoint(checkpoint_path):
                valid_checkpoints.append((checkpoint_path, metadata.creation_time))
        
        if not valid_checkpoints:
            logger.warning("No valid checkpoints found")
            return None
        
        # Sort by creation time (most recent first)
        valid_checkpoints.sort(key=lambda x: x[1], reverse=True)
        latest_checkpoint = valid_checkpoints[0][0]
        
        logger.info(f"Latest valid checkpoint: {latest_checkpoint}")
        return latest_checkpoint

    def find_best_checkpoint(self) -> Optional[Path]:
        """Find the checkpoint with best validation loss"""
        best_checkpoint = None
        best_loss = float('inf')
        
        for metadata in self.metadata.values():
            if metadata.validation_loss is not None and metadata.validation_loss < best_loss:
                checkpoint_path = Path(metadata.checkpoint_path)
                if self.validate_checkpoint(checkpoint_path):
                    best_checkpoint = checkpoint_path
                    best_loss = metadata.validation_loss
        
        if best_checkpoint:
            logger.info(f"Best checkpoint: {best_checkpoint} (loss: {best_loss:.4f})")
            
        return best_checkpoint

    def list_available_checkpoints(self) -> List[Tuple[Path, CheckpointMetadata]]:
        """List all available valid checkpoints"""
        available = []
        
        for metadata in self.metadata.values():
            checkpoint_path = Path(metadata.checkpoint_path)
            if self.validate_checkpoint(checkpoint_path):
                available.append((checkpoint_path, metadata))
        
        # Sort by creation time (most recent first)
        available.sort(key=lambda x: x[1].creation_time, reverse=True)
        
        return available

    def cleanup_old_checkpoints(self, force: bool = False):
        """Clean up old checkpoints based on retention policy"""
        self._cleanup_old_checkpoints(force)

    def _cleanup_old_checkpoints(self, force: bool = False):
        """Internal cleanup with retention policy"""
        if not force and len(self.metadata) <= self.keep_latest + self.keep_best:
            return
        
        # Check disk usage
        total_size = self._calculate_total_checkpoint_size()
        if total_size > self.max_disk_usage_gb * 1024 * 1024 * 1024:
            logger.warning(f"Checkpoint disk usage ({total_size/1024**3:.2f}GB) "
                          f"exceeds limit ({self.max_disk_usage_gb}GB)")
            force = True
        
        # Get all checkpoints sorted by creation time
        all_checkpoints = sorted(
            self.metadata.values(),
            key=lambda x: x.creation_time,
            reverse=True
        )
        
        # Identify checkpoints to keep
        keep_paths = set()
        
        # Keep latest N checkpoints
        for metadata in all_checkpoints[:self.keep_latest]:
            keep_paths.add(metadata.checkpoint_path)
        
        # Keep best N checkpoints
        best_checkpoints = sorted(
            [m for m in all_checkpoints if m.validation_loss is not None],
            key=lambda x: x.validation_loss
        )
        for metadata in best_checkpoints[:self.keep_best]:
            keep_paths.add(metadata.checkpoint_path)
        
        # Keep epoch interval checkpoints
        for metadata in all_checkpoints:
            if metadata.epoch % self.keep_epoch_interval == 0:
                keep_paths.add(metadata.checkpoint_path)
        
        # Remove checkpoints not in keep list
        removed_count = 0
        for checkpoint_path, metadata in list(self.metadata.items()):
            if checkpoint_path not in keep_paths:
                try:
                    path_obj = Path(checkpoint_path)
                    if path_obj.exists():
                        path_obj.unlink()
                    del self.metadata[checkpoint_path]
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old checkpoints")
            self._save_metadata()

    def get_checkpoint_info(self, checkpoint_path: Union[str, Path]) -> Optional[CheckpointMetadata]:
        """Get metadata for specific checkpoint"""
        return self._get_checkpoint_metadata(Path(checkpoint_path))

    def wait_for_pending_saves(self, timeout: float = 300.0):
        """Wait for all pending async saves to complete"""
        if not self._executor:
            return
        
        start_time = time.time()
        
        while self._active_saves and (time.time() - start_time) < timeout:
            self._cleanup_completed_saves()
            time.sleep(0.1)
        
        if self._active_saves:
            logger.warning(f"Timeout waiting for {len(self._active_saves)} pending saves")

    def emergency_save(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        scheduler_state: Dict[str, Any],
        epoch: int,
        step: Optional[int] = None,
        error_info: Optional[str] = None
    ) -> Path:
        """Emergency checkpoint save (always synchronous)"""
        logger.warning(f"Emergency checkpoint save triggered: {error_info}")
        
        return self.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            epoch=epoch,
            step=step,
            is_emergency=True,
            blocking=True
        )

    def _create_metadata(
        self,
        checkpoint_path: Path,
        checkpoint_data: Dict[str, Any],
        is_best: bool
    ) -> CheckpointMetadata:
        """Create metadata for checkpoint"""
        model_state = checkpoint_data['model_state_dict']
        
        # Calculate model parameters
        total_params = sum(
            param.numel() for param in model_state.values()
            if isinstance(param, torch.Tensor)
        )
        
        # Estimate model size
        model_size_mb = sum(
            param.numel() * param.element_size() for param in model_state.values()
            if isinstance(param, torch.Tensor)
        ) / (1024 * 1024)
        
        return CheckpointMetadata(
            checkpoint_path=str(checkpoint_path),
            epoch=checkpoint_data['epoch'],
            step=checkpoint_data.get('step'),
            timestamp=checkpoint_data['timestamp'],
            model_config=checkpoint_data.get('model_config', {}),
            training_metrics=checkpoint_data.get('metrics', {}),
            optimizer_type=type(checkpoint_data.get('optimizer_state_dict', {})).__name__,
            scheduler_type=type(checkpoint_data.get('scheduler_state_dict', {})).__name__,
            total_params=total_params,
            model_size_mb=model_size_mb,
            validation_loss=checkpoint_data.get('metrics', {}).get('val_loss'),
            is_best=is_best,
            resume_count=0,
            creation_time=time.time(),
            checksum="",  # Will be calculated after save
            compressed=self.enable_compression
        )

    def _update_metadata(self, metadata: CheckpointMetadata):
        """Update metadata tracking"""
        self.metadata[metadata.checkpoint_path] = metadata
        self._save_metadata()

    def _get_checkpoint_metadata(self, checkpoint_path: Path) -> Optional[CheckpointMetadata]:
        """Get metadata for checkpoint"""
        return self.metadata.get(str(checkpoint_path))

    def _load_metadata(self) -> Dict[str, CheckpointMetadata]:
        """Load checkpoint metadata from file"""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = {}
            for path, meta_dict in data.items():
                try:
                    metadata[path] = CheckpointMetadata.from_dict(meta_dict)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {path}: {e}")
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint metadata: {e}")
            return {}

    def _save_metadata(self):
        """Save checkpoint metadata to file"""
        try:
            metadata_dict = {
                path: metadata.to_dict()
                for path, metadata in self.metadata.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()

    def _calculate_total_checkpoint_size(self) -> int:
        """Calculate total size of all checkpoints in bytes"""
        total_size = 0
        
        for metadata in self.metadata.values():
            checkpoint_path = Path(metadata.checkpoint_path)
            if checkpoint_path.exists():
                total_size += checkpoint_path.stat().st_size
        
        return total_size

    def _cleanup_completed_saves(self):
        """Clean up completed async save futures"""
        completed = [f for f in self._active_saves if f.done()]
        
        for future in completed:
            self._active_saves.remove(future)
            try:
                future.result()  # This will raise if the save failed
            except Exception as e:
                logger.error(f"Async checkpoint save failed: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        if self._executor:
            self.wait_for_pending_saves()
            self._executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_executor') and self._executor:
            try:
                self._executor.shutdown(wait=False)
            except:
                pass


def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    **kwargs
) -> CheckpointManager:
    """Factory function to create checkpoint manager with sensible defaults"""
    return CheckpointManager(checkpoint_dir, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    import tempfile
    import torch.nn as nn
    
    # Create test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # Test checkpoint manager
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(temp_dir)
        
        model = TestModel()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Save test checkpoint
        checkpoint_path = manager.save_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=1,
            step=100,
            metrics={'loss': 0.5, 'val_loss': 0.6},
            blocking=True
        )
        
        print(f"Test checkpoint saved: {checkpoint_path}")
        
        # Load test checkpoint
        data, metadata = manager.load_checkpoint()
        print(f"Test checkpoint loaded: {metadata.checkpoint_path}")
        print(f"Metadata: epoch={metadata.epoch}, step={metadata.step}")
