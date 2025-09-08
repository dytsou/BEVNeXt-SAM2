#!/usr/bin/env python3
"""
Enhanced Resume System for Robust Training Continuation

This module provides a comprehensive resume system that addresses all common
issues with checkpoint loading and training state restoration.

Key Features:
- Complete training state preservation and restoration
- Device-aware checkpoint loading
- Configuration validation and compatibility checks
- Mixed precision scaler state handling
- Random seed state preservation
- Comprehensive error handling and validation
- Support for both simple and enhanced checkpoint systems

Author: Senior Python Programmer & AI Training Expert
"""

import os
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ResumeResult:
    """Result of a resume operation"""
    success: bool
    checkpoint_path: Optional[Path]
    restored_epoch: Optional[int]
    restored_step: Optional[int]
    message: str
    warnings: List[str]
    metadata: Optional[Dict[str, Any]]


class EnhancedResumeManager:
    """
    Enhanced resume manager that handles complete training state restoration
    with robust error handling and validation
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        output_dir: Union[str, Path],
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        config: Optional[Dict[str, Any]] = None,
        strict_config_check: bool = True,
        allow_partial_load: bool = False
    ):
        """
        Initialize enhanced resume manager
        
        Args:
            model: PyTorch model to restore
            optimizer: Optimizer to restore
            scheduler: Learning rate scheduler to restore
            device: Target device for restoration
            output_dir: Directory containing checkpoints
            scaler: Mixed precision scaler (if used)
            config: Current training configuration
            strict_config_check: Whether to strictly validate config compatibility
            allow_partial_load: Whether to allow partial model loading
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.scaler = scaler
        self.config = config
        self.strict_config_check = strict_config_check
        self.allow_partial_load = allow_partial_load
        
        # State tracking
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_stats = defaultdict(list)
        self.batch_step = 0
        
        logger.info(f"Enhanced resume manager initialized for: {self.output_dir}")
    
    def find_resumable_checkpoints(self) -> List[Path]:
        """
        Find all potentially resumable checkpoints in the output directory
        
        Returns:
            List of checkpoint paths sorted by modification time (newest first)
        """
        checkpoint_patterns = [
            '*.pth',
            '*.pt',
            'checkpoint_*.pth',
            'checkpoint_*.pt',
            'model_*.pth',
            'model_*.pt'
        ]
        
        found_checkpoints = []
        
        # Search in output directory and subdirectories
        for pattern in checkpoint_patterns:
            found_checkpoints.extend(self.output_dir.glob(pattern))
            found_checkpoints.extend(self.output_dir.glob(f'**/{pattern}'))
        
        # Remove duplicates
        unique_checkpoints = list(set(found_checkpoints))
        
        # Enhanced sorting: prioritize by training progress (epoch + step)
        sorted_checkpoints = self._sort_checkpoints_by_progress(unique_checkpoints)
        
        logger.info(f"Found {len(sorted_checkpoints)} potential checkpoint files")
        for i, ckpt in enumerate(sorted_checkpoints[:5]):  # Log top 5
            mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime)
            epoch, step = self._extract_epoch_step_from_filename(ckpt)
            progress_info = f"epoch {epoch}, step {step}" if epoch is not None else "unknown progress"
            logger.info(f"  {i+1}. {ckpt.name} ({progress_info}, modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
        
        return sorted_checkpoints
    
    def _extract_epoch_step_from_filename(self, checkpoint_path: Path) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract epoch and step information from checkpoint filename
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Tuple of (epoch, step) or (None, None) if not extractable
        """
        import re
        
        filename = checkpoint_path.name
        
        # Pattern for sub-epoch checkpoints: checkpoint_epoch_X_step_Y.pth
        step_pattern = r'checkpoint_epoch_(\d+)_step_(\d+)\.pth'
        step_match = re.search(step_pattern, filename)
        if step_match:
            epoch = int(step_match.group(1))
            step = int(step_match.group(2))
            return epoch, step
        
        # Pattern for epoch checkpoints: checkpoint_epoch_X.pth
        epoch_pattern = r'checkpoint_epoch_(\d+)\.pth'
        epoch_match = re.search(epoch_pattern, filename)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            return epoch, 0  # Start of epoch
        
        # Special checkpoints
        if 'latest' in filename.lower():
            return 999999, 999999  # High priority for latest
        elif 'best' in filename.lower():
            return 999998, 999999  # High priority for best
        
        return None, None
    
    def _sort_checkpoints_by_progress(self, checkpoints: List[Path]) -> List[Path]:
        """
        Sort checkpoints by training progress (epoch, then step)
        
        Args:
            checkpoints: List of checkpoint paths
            
        Returns:
            Sorted list with most advanced training first
        """
        def progress_key(checkpoint_path: Path) -> Tuple[int, int, float]:
            epoch, step = self._extract_epoch_step_from_filename(checkpoint_path)
            
            # If we can't extract epoch/step, fall back to modification time
            if epoch is None:
                mod_time = checkpoint_path.stat().st_mtime
                return (0, 0, mod_time)
            
            # Primary sort: epoch (descending), secondary: step (descending), tertiary: mod time (descending)
            mod_time = checkpoint_path.stat().st_mtime
            return (epoch, step or 0, mod_time)
        
        # Sort by progress (most advanced first)
        sorted_checkpoints = sorted(checkpoints, key=progress_key, reverse=True)
        
        logger.debug(f"Checkpoint sorting order:")
        for i, ckpt in enumerate(sorted_checkpoints[:10]):  # Log top 10
            epoch, step = self._extract_epoch_step_from_filename(ckpt)
            logger.debug(f"  {i+1}. {ckpt.name} (epoch: {epoch}, step: {step})")
        
        return sorted_checkpoints
    
    def auto_resume(self) -> ResumeResult:
        """
        Automatically find and resume from the best available checkpoint
        
        Returns:
            ResumeResult indicating success/failure and details
        """
        logger.info("🔄 Starting automatic resume detection...")
        
        # Find potential checkpoints
        checkpoints = self.find_resumable_checkpoints()
        
        if not checkpoints:
            return ResumeResult(
                success=False,
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message="No checkpoint files found for resume",
                warnings=[],
                metadata=None
            )
        
        # Try to resume from each checkpoint (newest first)
        for checkpoint_path in checkpoints:
            logger.info(f"Attempting to resume from: {checkpoint_path}")
            
            result = self.resume_from_checkpoint(checkpoint_path)
            if result.success:
                logger.info(f"✅ Successfully resumed from: {checkpoint_path}")
                return result
            else:
                logger.warning(f"❌ Failed to resume from {checkpoint_path}: {result.message}")
        
        # If all checkpoints failed, return the last result
        return ResumeResult(
            success=False,
            checkpoint_path=None,
            restored_epoch=None,
            restored_step=None,
            message=f"Failed to resume from all {len(checkpoints)} available checkpoints",
            warnings=[f"Tried {len(checkpoints)} checkpoints"],
            metadata=None
        )
    
    def resume_from_checkpoint(
        self, 
        checkpoint_path: Union[str, Path],
        force_resume: bool = False
    ) -> ResumeResult:
        """
        Resume training from a specific checkpoint with comprehensive validation
        
        Args:
            checkpoint_path: Path to checkpoint file
            force_resume: Skip some validation checks if True
            
        Returns:
            ResumeResult indicating success/failure and details
        """
        checkpoint_path = Path(checkpoint_path)
        warnings_list = []
        
        if not checkpoint_path.exists():
            return ResumeResult(
                success=False,
                checkpoint_path=checkpoint_path,
                restored_epoch=None,
                restored_step=None,
                message=f"Checkpoint file not found: {checkpoint_path}",
                warnings=[],
                metadata=None
            )
        
        try:
            # Load checkpoint with proper device mapping
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            start_time = time.time()
            
            # Load to CPU first, then move to target device
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            load_time = time.time() - start_time
            
            logger.info(f"Checkpoint loaded in {load_time:.2f} seconds")
            
            # Validate checkpoint structure
            validation_result = self._validate_checkpoint_structure(checkpoint_data, checkpoint_path)
            if not validation_result.success and not force_resume:
                return validation_result
            warnings_list.extend(validation_result.warnings)
            
            # Validate configuration compatibility
            if self.config and not force_resume:
                config_result = self._validate_config_compatibility(checkpoint_data)
                if not config_result.success and self.strict_config_check:
                    return config_result
                warnings_list.extend(config_result.warnings)
            
            # Restore model state
            model_result = self._restore_model_state(checkpoint_data)
            if not model_result.success:
                return model_result
            warnings_list.extend(model_result.warnings)
            
            # Restore optimizer state  
            optimizer_result = self._restore_optimizer_state(checkpoint_data)
            if not optimizer_result.success and not force_resume:
                return optimizer_result
            warnings_list.extend(optimizer_result.warnings)
            
            # Restore scheduler state
            scheduler_result = self._restore_scheduler_state(checkpoint_data)
            if not scheduler_result.success and not force_resume:
                return scheduler_result
            warnings_list.extend(scheduler_result.warnings)
            
            # Restore mixed precision scaler state
            if self.scaler:
                scaler_result = self._restore_scaler_state(checkpoint_data)
                warnings_list.extend(scaler_result.warnings)
            
            # Restore training state
            training_state_result = self._restore_training_state(checkpoint_data)
            warnings_list.extend(training_state_result.warnings)
            
            # Restore random states
            random_state_result = self._restore_random_states(checkpoint_data)
            warnings_list.extend(random_state_result.warnings)
            
            # Log successful resume
            logger.info(f"✅ Resume successful!")
            logger.info(f"   📊 Restored to epoch: {self.epoch}")
            logger.info(f"   📈 Best validation loss: {self.best_val_loss:.6f}")
            logger.info(f"   🔢 Training step: {self.batch_step}")
            
            if warnings_list:
                logger.warning(f"   ⚠️  Resume completed with {len(warnings_list)} warnings")
                for warning in warnings_list:
                    logger.warning(f"      - {warning}")
            
            return ResumeResult(
                success=True,
                checkpoint_path=checkpoint_path,
                restored_epoch=self.epoch,
                restored_step=self.batch_step,
                message=f"Successfully resumed from epoch {self.epoch}",
                warnings=warnings_list,
                metadata=self._extract_checkpoint_metadata(checkpoint_data)
            )
            
        except Exception as e:
            logger.error(f"❌ Resume failed with exception: {e}")
            return ResumeResult(
                success=False,
                checkpoint_path=checkpoint_path,
                restored_epoch=None,
                restored_step=None,
                message=f"Resume failed with error: {str(e)}",
                warnings=warnings_list,
                metadata=None
            )
    
    def _validate_checkpoint_structure(self, checkpoint_data: Dict, checkpoint_path: Path) -> ResumeResult:
        """Validate that checkpoint has required structure"""
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        optional_keys = ['scheduler_state_dict', 'scaler_state_dict', 'config', 'training_stats']
        warnings_list = []
        
        # Check required keys
        missing_keys = [key for key in required_keys if key not in checkpoint_data]
        if missing_keys:
            return ResumeResult(
                success=False,
                checkpoint_path=checkpoint_path,
                restored_epoch=None,
                restored_step=None,
                message=f"Checkpoint missing required keys: {missing_keys}",
                warnings=[],
                metadata=None
            )
        
        # Check optional keys
        missing_optional = [key for key in optional_keys if key not in checkpoint_data]
        if missing_optional:
            warnings_list.append(f"Missing optional keys: {missing_optional}")
        
        # Validate epoch value
        try:
            epoch = int(checkpoint_data['epoch'])
            if epoch < 0:
                warnings_list.append(f"Invalid epoch value: {epoch}")
        except (ValueError, TypeError):
            return ResumeResult(
                success=False,
                checkpoint_path=checkpoint_path,
                restored_epoch=None,
                restored_step=None,
                message=f"Invalid epoch value in checkpoint: {checkpoint_data.get('epoch')}",
                warnings=warnings_list,
                metadata=None
            )
        
        return ResumeResult(
            success=True,
            checkpoint_path=checkpoint_path,
            restored_epoch=epoch,
            restored_step=None,
            message="Checkpoint structure validation passed",
            warnings=warnings_list,
            metadata=None
        )
    
    def _validate_config_compatibility(self, checkpoint_data: Dict) -> ResumeResult:
        """Validate configuration compatibility"""
        warnings_list = []
        
        if 'config' not in checkpoint_data:
            warnings_list.append("No configuration found in checkpoint for compatibility check")
            return ResumeResult(
                success=True,
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message="Config validation skipped - no config in checkpoint",
                warnings=warnings_list,
                metadata=None
            )
        
        checkpoint_config = checkpoint_data['config']
        
        # Check critical configuration parameters
        critical_params = ['d_model', 'num_classes', 'num_queries', 'bev_size']
        
        for param in critical_params:
            if param in self.config and param in checkpoint_config:
                if self.config[param] != checkpoint_config[param]:
                    if self.strict_config_check:
                        return ResumeResult(
                            success=False,
                            checkpoint_path=None,
                            restored_epoch=None,
                            restored_step=None,
                            message=f"Config mismatch for {param}: current={self.config[param]}, checkpoint={checkpoint_config[param]}",
                            warnings=warnings_list,
                            metadata=None
                        )
                    else:
                        warnings_list.append(f"Config mismatch for {param}: current={self.config[param]}, checkpoint={checkpoint_config[param]}")
        
        return ResumeResult(
            success=True,
            checkpoint_path=None,
            restored_epoch=None,
            restored_step=None,
            message="Configuration compatibility validated",
            warnings=warnings_list,
            metadata=None
        )
    
    def _restore_model_state(self, checkpoint_data: Dict) -> ResumeResult:
        """Restore model state with proper error handling"""
        warnings_list = []
        
        try:
            model_state = checkpoint_data['model_state_dict']
            
            # Handle DataParallel/DistributedDataParallel wrappers
            if hasattr(self.model, 'module'):
                # Model is wrapped, but checkpoint might not be
                try:
                    self.model.load_state_dict(model_state, strict=not self.allow_partial_load)
                except RuntimeError:
                    # Try loading into the wrapped module
                    self.model.module.load_state_dict(model_state, strict=not self.allow_partial_load)
                    warnings_list.append("Loaded state dict into wrapped model module")
            else:
                # Model is not wrapped, but checkpoint might be from wrapped model
                try:
                    self.model.load_state_dict(model_state, strict=not self.allow_partial_load)
                except RuntimeError as e:
                    if 'module.' in str(e):
                        # Try removing 'module.' prefix from checkpoint keys
                        unwrapped_state = {}
                        for key, value in model_state.items():
                            new_key = key.replace('module.', '') if key.startswith('module.') else key
                            unwrapped_state[new_key] = value
                        
                        self.model.load_state_dict(unwrapped_state, strict=not self.allow_partial_load)
                        warnings_list.append("Removed 'module.' prefix from checkpoint keys")
                    else:
                        raise
            
            # Move model to target device
            self.model.to(self.device)
            
            return ResumeResult(
                success=True,
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message="Model state restored successfully",
                warnings=warnings_list,
                metadata=None
            )
            
        except Exception as e:
            return ResumeResult(
                success=False,
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message=f"Failed to restore model state: {str(e)}",
                warnings=warnings_list,
                metadata=None
            )
    
    def _restore_optimizer_state(self, checkpoint_data: Dict) -> ResumeResult:
        """Restore optimizer state with device handling"""
        warnings_list = []
        
        try:
            if 'optimizer_state_dict' not in checkpoint_data:
                warnings_list.append("No optimizer state found in checkpoint")
                return ResumeResult(
                    success=True,
                    checkpoint_path=None,
                    restored_epoch=None,
                    restored_step=None,
                    message="Optimizer state restore skipped",
                    warnings=warnings_list,
                    metadata=None
                )
            
            optimizer_state = checkpoint_data['optimizer_state_dict']
            
            # Load optimizer state
            self.optimizer.load_state_dict(optimizer_state)
            
            # Move optimizer states to target device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            
            return ResumeResult(
                success=True,
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message="Optimizer state restored successfully",
                warnings=warnings_list,
                metadata=None
            )
            
        except Exception as e:
            return ResumeResult(
                success=False,
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message=f"Failed to restore optimizer state: {str(e)}",
                warnings=warnings_list,
                metadata=None
            )
    
    def _restore_scheduler_state(self, checkpoint_data: Dict) -> ResumeResult:
        """Restore learning rate scheduler state"""
        warnings_list = []
        
        try:
            if 'scheduler_state_dict' not in checkpoint_data:
                warnings_list.append("No scheduler state found in checkpoint")
                return ResumeResult(
                    success=True,
                    checkpoint_path=None,
                    restored_epoch=None,
                    restored_step=None,
                    message="Scheduler state restore skipped",
                    warnings=warnings_list,
                    metadata=None
                )
            
            scheduler_state = checkpoint_data['scheduler_state_dict']
            self.scheduler.load_state_dict(scheduler_state)
            
            return ResumeResult(
                success=True,
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message="Scheduler state restored successfully",
                warnings=warnings_list,
                metadata=None
            )
            
        except Exception as e:
            warnings_list.append(f"Failed to restore scheduler state: {str(e)}")
            return ResumeResult(
                success=True,  # Non-critical failure
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message="Scheduler state restore failed (non-critical)",
                warnings=warnings_list,
                metadata=None
            )
    
    def _restore_scaler_state(self, checkpoint_data: Dict) -> ResumeResult:
        """Restore mixed precision scaler state"""
        warnings_list = []
        
        try:
            if 'scaler_state_dict' not in checkpoint_data:
                warnings_list.append("No scaler state found in checkpoint")
                return ResumeResult(
                    success=True,
                    checkpoint_path=None,
                    restored_epoch=None,
                    restored_step=None,
                    message="Scaler state restore skipped",
                    warnings=warnings_list,
                    metadata=None
                )
            
            scaler_state = checkpoint_data['scaler_state_dict']
            self.scaler.load_state_dict(scaler_state)
            
            return ResumeResult(
                success=True,
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message="Scaler state restored successfully",
                warnings=warnings_list,
                metadata=None
            )
            
        except Exception as e:
            warnings_list.append(f"Failed to restore scaler state: {str(e)}")
            return ResumeResult(
                success=True,  # Non-critical failure
                checkpoint_path=None,
                restored_epoch=None,
                restored_step=None,
                message="Scaler state restore failed (non-critical)",
                warnings=warnings_list,
                metadata=None
            )
    
    def _restore_training_state(self, checkpoint_data: Dict) -> ResumeResult:
        """Restore training state variables"""
        warnings_list = []
        
        # Restore epoch
        self.epoch = checkpoint_data.get('epoch', 0)
        
        # Restore best validation loss
        self.best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
        
        # Restore batch step
        self.batch_step = checkpoint_data.get('batch_step', 0)
        
        # Restore training statistics
        if 'training_stats' in checkpoint_data:
            try:
                self.training_stats = defaultdict(list, checkpoint_data['training_stats'])
            except Exception as e:
                warnings_list.append(f"Failed to restore training statistics: {e}")
        else:
            warnings_list.append("No training statistics found in checkpoint")
        
        return ResumeResult(
            success=True,
            checkpoint_path=None,
            restored_epoch=self.epoch,
            restored_step=self.batch_step,
            message="Training state restored successfully",
            warnings=warnings_list,
            metadata=None
        )
    
    def _restore_random_states(self, checkpoint_data: Dict) -> ResumeResult:
        """Restore random number generator states for reproducibility"""
        warnings_list = []
        
        try:
            # Restore PyTorch random state
            if 'torch_rng_state' in checkpoint_data:
                torch.set_rng_state(checkpoint_data['torch_rng_state'])
            else:
                warnings_list.append("No PyTorch RNG state found in checkpoint")
            
            # Restore CUDA random state  
            if 'cuda_rng_state' in checkpoint_data and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint_data['cuda_rng_state'])
            else:
                if torch.cuda.is_available():
                    warnings_list.append("No CUDA RNG state found in checkpoint")
            
            # Restore NumPy random state
            if 'numpy_rng_state' in checkpoint_data:
                np.random.set_state(checkpoint_data['numpy_rng_state'])
            else:
                warnings_list.append("No NumPy RNG state found in checkpoint")
            
            # Restore Python random state
            if 'python_rng_state' in checkpoint_data:
                import random
                random.setstate(checkpoint_data['python_rng_state'])
            else:
                warnings_list.append("No Python RNG state found in checkpoint")
                
        except Exception as e:
            warnings_list.append(f"Failed to restore random states: {e}")
        
        return ResumeResult(
            success=True,
            checkpoint_path=None,
            restored_epoch=None,
            restored_step=None,
            message="Random states restoration attempted",
            warnings=warnings_list,
            metadata=None
        )
    
    def _extract_checkpoint_metadata(self, checkpoint_data: Dict) -> Dict[str, Any]:
        """Extract metadata from checkpoint for reporting"""
        metadata = {}
        
        # Basic training info
        metadata['epoch'] = checkpoint_data.get('epoch', 0)
        metadata['step'] = checkpoint_data.get('batch_step', 0)
        metadata['best_val_loss'] = checkpoint_data.get('best_val_loss', float('inf'))
        
        # Model info
        if 'model_state_dict' in checkpoint_data:
            model_params = sum(
                p.numel() for p in checkpoint_data['model_state_dict'].values()
                if isinstance(p, torch.Tensor)
            )
            metadata['model_parameters'] = model_params
        
        # Training statistics
        if 'training_stats' in checkpoint_data:
            stats = checkpoint_data['training_stats']
            if isinstance(stats, dict):
                metadata['training_epochs'] = len(stats.get('train_total', []))
                if 'val_total' in stats and stats['val_total']:
                    metadata['best_recorded_val_loss'] = min(stats['val_total'])
        
        # Configuration
        if 'config' in checkpoint_data:
            config = checkpoint_data['config']
            metadata['model_config'] = {
                'd_model': config.get('d_model'),
                'num_classes': config.get('num_classes'),
                'batch_size': config.get('batch_size'),
                'learning_rate': config.get('learning_rate')
            }
        
        return metadata
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for external access"""
        return {
            'epoch': self.epoch,
            'batch_step': self.batch_step,
            'best_val_loss': self.best_val_loss,
            'training_stats': dict(self.training_stats)
        }
    
    def create_enhanced_checkpoint(
        self,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive checkpoint with all necessary state
        
        Args:
            additional_data: Additional data to include in checkpoint
            
        Returns:
            Complete checkpoint dictionary
        """
        checkpoint = {
            # Core model and training state
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            
            # Training progress
            'epoch': self.epoch,
            'batch_step': self.batch_step,
            'best_val_loss': self.best_val_loss,
            'training_stats': dict(self.training_stats),
            
            # Configuration
            'config': self.config,
            
            # Metadata
            'timestamp': datetime.now().isoformat(),
            'save_device': str(self.device),
            
            # Random states for reproducibility
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
        }
        
        # Add Python random state
        try:
            import random
            checkpoint['python_rng_state'] = random.getstate()
        except:
            pass
        
        # Add CUDA random state if available
        if torch.cuda.is_available():
            try:
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
            except:
                pass
        
        # Add mixed precision scaler state
        if self.scaler:
            try:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            except:
                pass
        
        # Add any additional data
        if additional_data:
            checkpoint.update(additional_data)
        
        return checkpoint


def create_enhanced_resume_manager(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    output_dir: Union[str, Path],
    **kwargs
) -> EnhancedResumeManager:
    """
    Factory function to create enhanced resume manager
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler  
        device: Target device
        output_dir: Checkpoint directory
        **kwargs: Additional arguments for EnhancedResumeManager
        
    Returns:
        Configured EnhancedResumeManager instance
    """
    return EnhancedResumeManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    import tempfile
    import torch.nn as nn
    
    # Create test model and components
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # Test enhanced resume manager
    with tempfile.TemporaryDirectory() as temp_dir:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TestModel().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Create resume manager
        resume_manager = create_enhanced_resume_manager(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=temp_dir,
            config={'d_model': 512, 'num_classes': 10}
        )
        
        # Test checkpoint creation
        checkpoint = resume_manager.create_enhanced_checkpoint()
        checkpoint_path = Path(temp_dir) / 'test_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        print(f"Test checkpoint saved: {checkpoint_path}")
        
        # Test resume
        resume_manager.epoch = 5  # Simulate some training
        result = resume_manager.resume_from_checkpoint(checkpoint_path)
        
        print(f"Resume result: {result.success}")
        print(f"Message: {result.message}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        if result.metadata:
            print(f"Metadata: {result.metadata}")
