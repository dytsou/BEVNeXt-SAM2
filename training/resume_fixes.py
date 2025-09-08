#!/usr/bin/env python3
"""
Resume Functionality Fixes and Enhancements

This module provides fixes for the training resume functionality to ensure
proper continuation of training from checkpoints.

Key fixes:
1. Proper batch_step tracking across epochs
2. Resume from exact batch position within an epoch
3. Consistent checkpoint data format
4. Dataset state preservation

Author: Senior Python Programmer & AI Training Expert
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

logger = logging.getLogger(__name__)


class EnhancedTrainingState:
    """Enhanced training state tracker that properly handles resume"""
    
    def __init__(self):
        self.epoch = 0
        self.global_step = 0  # Total batches processed across all epochs
        self.epoch_step = 0   # Batches processed in current epoch
        self.best_val_loss = float('inf')
        self.training_stats = defaultdict(list)
        self.total_epochs = 0
        self.batches_per_epoch = 0
        self.resume_info = {}
        
    def update_step(self, batch_idx: int, epoch: int):
        """Update step counters"""
        self.epoch = epoch
        self.epoch_step = batch_idx
        self.global_step += 1
        
    def should_skip_batch(self, batch_idx: int) -> bool:
        """Check if batch should be skipped during resume"""
        if self.resume_info.get('resumed_epoch') == self.epoch:
            return batch_idx < self.resume_info.get('resumed_epoch_step', 0)
        return False
        
    def save_state(self) -> Dict[str, Any]:
        """Save complete training state"""
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'epoch_step': self.epoch_step,
            'best_val_loss': self.best_val_loss,
            'training_stats': dict(self.training_stats),
            'total_epochs': self.total_epochs,
            'batches_per_epoch': self.batches_per_epoch,
            'timestamp': datetime.now().isoformat()
        }
        
    def load_state(self, state_dict: Dict[str, Any]):
        """Load training state from checkpoint"""
        self.epoch = state_dict.get('epoch', 0)
        self.global_step = state_dict.get('global_step', 0)
        self.epoch_step = state_dict.get('epoch_step', 0)
        self.best_val_loss = state_dict.get('best_val_loss', float('inf'))
        self.training_stats = defaultdict(list, state_dict.get('training_stats', {}))
        self.total_epochs = state_dict.get('total_epochs', 0)
        self.batches_per_epoch = state_dict.get('batches_per_epoch', 0)
        
        # Set resume info
        self.resume_info = {
            'resumed_epoch': self.epoch,
            'resumed_epoch_step': self.epoch_step,
            'resumed_global_step': self.global_step,
            'resume_timestamp': datetime.now().isoformat()
        }


def create_resume_compatible_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    training_state: EnhancedTrainingState,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a checkpoint with all necessary information for proper resume
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        training_state: Enhanced training state tracker
        scaler: Mixed precision scaler (optional)
        additional_data: Additional data to include in checkpoint
        
    Returns:
        Complete checkpoint dictionary
    """
    checkpoint = {
        # Model and optimizer states
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        
        # Training state - using consistent naming
        'epoch': training_state.epoch,
        'step': training_state.global_step,  # For compatibility with checkpoint_manager
        'batch_step': training_state.global_step,  # For compatibility with enhanced_resume
        'epoch_step': training_state.epoch_step,  # New: track position within epoch
        'global_step': training_state.global_step,  # New: total steps
        
        # Training progress
        'best_val_loss': training_state.best_val_loss,
        'training_stats': dict(training_state.training_stats),
        'training_state': training_state.save_state(),  # Complete state backup
        
        # Metadata
        'timestamp': datetime.now().isoformat(),
        'checkpoint_version': '2.0',  # Version for compatibility checking
        
        # Random states for reproducibility
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': None,  # Will be set if numpy is available
        'python_rng_state': None,  # Will be set if random is available
    }
    
    # Add CUDA random state if available
    if torch.cuda.is_available():
        try:
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
            checkpoint['cuda_rng_state_all'] = [torch.cuda.get_rng_state(i) 
                                                for i in range(torch.cuda.device_count())]
        except:
            pass
    
    # Add numpy random state
    try:
        import numpy as np
        checkpoint['numpy_rng_state'] = np.random.get_state()
    except:
        pass
    
    # Add python random state
    try:
        import random
        checkpoint['python_rng_state'] = random.getstate()
    except:
        pass
    
    # Add mixed precision scaler state
    if scaler is not None:
        try:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        except:
            pass
    
    # Add any additional data
    if additional_data:
        checkpoint.update(additional_data)
    
    return checkpoint


def load_checkpoint_with_compatibility(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    training_state: EnhancedTrainingState,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Load checkpoint with backward compatibility and proper state restoration
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to restore
        optimizer: Optimizer to restore
        scheduler: Scheduler to restore
        training_state: Training state to restore
        device: Target device
        scaler: Mixed precision scaler (optional)
        strict: Whether to use strict model loading
        
    Returns:
        Tuple of (success, warnings)
    """
    warnings = []
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check version compatibility
        checkpoint_version = checkpoint.get('checkpoint_version', '1.0')
        if checkpoint_version != '2.0':
            warnings.append(f"Loading older checkpoint version {checkpoint_version}")
        
        # Load model state
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        except RuntimeError as e:
            if 'module.' in str(e):
                # Handle DataParallel/DistributedDataParallel mismatch
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '') if key.startswith('module.') else key
                    new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=strict)
                warnings.append("Removed 'module.' prefix from model state dict keys")
            else:
                raise
        
        # Move model to device
        model.to(device)
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer states to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        else:
            warnings.append("No optimizer state found in checkpoint")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            warnings.append("No scheduler state found in checkpoint")
        
        # Load training state with compatibility handling
        if 'training_state' in checkpoint:
            # New format with complete state
            training_state.load_state(checkpoint['training_state'])
        else:
            # Fallback to old format
            state_dict = {
                'epoch': checkpoint.get('epoch', 0),
                'global_step': checkpoint.get('global_step', 
                                           checkpoint.get('step', 
                                           checkpoint.get('batch_step', 0))),
                'epoch_step': checkpoint.get('epoch_step', 0),
                'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
                'training_stats': checkpoint.get('training_stats', {}),
            }
            training_state.load_state(state_dict)
            warnings.append("Loaded training state from legacy checkpoint format")
        
        # Load scaler state if available
        if scaler and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except:
                warnings.append("Failed to load scaler state")
        
        # Restore random states
        if 'torch_rng_state' in checkpoint:
            try:
                torch.set_rng_state(checkpoint['torch_rng_state'])
            except:
                warnings.append("Failed to restore PyTorch random state")
        
        if 'cuda_rng_state_all' in checkpoint and torch.cuda.is_available():
            try:
                for i, state in enumerate(checkpoint['cuda_rng_state_all']):
                    if i < torch.cuda.device_count():
                        torch.cuda.set_rng_state(state, i)
            except:
                warnings.append("Failed to restore CUDA random states")
        
        logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resumed at epoch {training_state.epoch}, global step {training_state.global_step}")
        
        return True, warnings
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False, [f"Checkpoint loading failed: {str(e)}"]


class ResumeAwareDataLoader:
    """DataLoader wrapper that can skip batches for resume"""
    
    def __init__(self, dataloader, training_state: EnhancedTrainingState):
        self.dataloader = dataloader
        self.training_state = training_state
        self.epoch = training_state.epoch
        
    def __iter__(self):
        for batch_idx, batch in enumerate(self.dataloader):
            # Skip batches if resuming within an epoch
            if self.training_state.should_skip_batch(batch_idx):
                logger.debug(f"Skipping batch {batch_idx} (already processed)")
                continue
            yield batch_idx, batch
            
    def __len__(self):
        return len(self.dataloader)


def create_resume_manager_with_fixes(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    output_dir: Union[str, Path],
    **kwargs
) -> 'FixedEnhancedResumeManager':
    """
    Create a fixed version of the enhanced resume manager
    
    This factory function creates a resume manager with all the fixes
    for proper training continuation.
    """
    from training.enhanced_resume import EnhancedResumeManager
    
    class FixedEnhancedResumeManager(EnhancedResumeManager):
        """Fixed version with proper batch tracking"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training_state = EnhancedTrainingState()
            
        def create_enhanced_checkpoint(self, additional_data=None):
            """Override to use fixed checkpoint creation"""
            return create_resume_compatible_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                training_state=self.training_state,
                scaler=self.scaler,
                additional_data=additional_data
            )
            
        def resume_from_checkpoint(self, checkpoint_path, force_resume=False):
            """Override to use fixed checkpoint loading"""
            success, warnings = load_checkpoint_with_compatibility(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                training_state=self.training_state,
                device=self.device,
                scaler=self.scaler,
                strict=not self.allow_partial_load
            )
            
            if success:
                # Update instance attributes from training state
                self.epoch = self.training_state.epoch
                self.best_val_loss = self.training_state.best_val_loss
                self.training_stats = self.training_state.training_stats
                self.batch_step = self.training_state.global_step
                
                return self._create_success_result(checkpoint_path, warnings)
            else:
                return self._create_failure_result(checkpoint_path, warnings)
                
        def _create_success_result(self, checkpoint_path, warnings):
            """Create successful resume result"""
            from training.enhanced_resume import ResumeResult
            return ResumeResult(
                success=True,
                checkpoint_path=Path(checkpoint_path),
                restored_epoch=self.training_state.epoch,
                restored_step=self.training_state.global_step,
                message=f"Successfully resumed from epoch {self.training_state.epoch}",
                warnings=warnings,
                metadata={'training_state': self.training_state.save_state()}
            )
            
        def _create_failure_result(self, checkpoint_path, warnings):
            """Create failed resume result"""
            from training.enhanced_resume import ResumeResult
            return ResumeResult(
                success=False,
                checkpoint_path=Path(checkpoint_path),
                restored_epoch=None,
                restored_step=None,
                message=f"Failed to resume from {checkpoint_path}",
                warnings=warnings,
                metadata=None
            )
    
    return FixedEnhancedResumeManager(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        **kwargs
    )


if __name__ == "__main__":
    # Test the fixes
    import tempfile
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
    # Create test components
    model = DummyModel()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
    device = torch.device('cpu')
    
    # Create training state
    training_state = EnhancedTrainingState()
    training_state.epoch = 5
    training_state.global_step = 523
    training_state.epoch_step = 23
    training_state.batches_per_epoch = 100
    
    # Create checkpoint
    checkpoint = create_resume_compatible_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state
    )
    
    print("Created checkpoint with keys:", list(checkpoint.keys()))
    print(f"Training state: epoch={checkpoint['epoch']}, global_step={checkpoint['global_step']}")
    
    # Test loading
    with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
        torch.save(checkpoint, tmp.name)
        
        # Reset training state
        new_training_state = EnhancedTrainingState()
        
        # Load checkpoint
        success, warnings = load_checkpoint_with_compatibility(
            checkpoint_path=tmp.name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=new_training_state,
            device=device
        )
        
        print(f"Load success: {success}")
        print(f"Warnings: {warnings}")
        print(f"Restored state: epoch={new_training_state.epoch}, "
              f"global_step={new_training_state.global_step}, "
              f"epoch_step={new_training_state.epoch_step}")
