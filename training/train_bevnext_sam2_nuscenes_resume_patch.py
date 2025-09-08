#!/usr/bin/env python3
"""
Patch for train_bevnext_sam2_nuscenes.py to fix resume functionality

This module shows the modifications needed to properly implement resume
functionality in the NuScenes training script.

Author: Senior Python Programmer & AI Training Expert
"""

# The following modifications should be applied to train_bevnext_sam2_nuscenes.py:

# 1. Import the fixes module
from training.resume_fixes import (
    EnhancedTrainingState, 
    create_resume_compatible_checkpoint,
    load_checkpoint_with_compatibility,
    ResumeAwareDataLoader
)

# 2. Modify the NuScenesTrainer class __init__ method
def __init__(self, config, use_mixed_precision=True, gpu_ids=None, 
             distributed=False, gradient_accumulation=1, lr_scaling=False,
             auto_resume=True, checkpoint_freq=100, no_resume_prompt=False,
             checkpoint_validation=True, resume_from=None):
    """Initialize enhanced NuScenes trainer"""
    # ... existing initialization code ...
    
    # Replace simple state tracking with EnhancedTrainingState
    self.training_state = EnhancedTrainingState()
    self.epoch = 0  # Keep for backward compatibility
    self.best_val_loss = float('inf')  # Keep for backward compatibility
    
    # ... rest of initialization ...


# 3. Modify the _train_epoch_enhanced method
def _train_epoch_enhanced(self):
    """Enhanced training epoch with proper step tracking and sub-epoch checkpointing"""
    self.model.train()
    epoch_losses = defaultdict(float)
    epoch_metrics = defaultdict(float)
    
    # Create resume-aware dataloader
    resume_aware_loader = ResumeAwareDataLoader(self.train_loader, self.training_state)
    
    # Setup progress bar (only for main process)
    if self.gpu_wrapper.is_main_process():
        # Account for skipped batches in progress bar
        initial = self.training_state.epoch_step if self.training_state.resume_info.get('resumed_epoch') == self.epoch else 0
        pbar = tqdm(resume_aware_loader, desc=f'Epoch {self.epoch+1} Train', 
                   initial=initial, total=len(self.train_loader))
    else:
        pbar = resume_aware_loader
    
    # Track if this is a resumed epoch
    is_resumed_epoch = self.training_state.resume_info.get('resumed_epoch') == self.epoch

    for batch_idx, batch in pbar:
        try:
            # Update training state - this now properly tracks global steps
            self.training_state.update_step(batch_idx, self.epoch)
            self.batch_step = self.training_state.global_step  # For compatibility
            
            # Training step (enhanced or standard based on configuration)
            if self.enhanced_checkpoints:
                losses, predictions = self._enhanced_training_step_with_checkpointing(batch, batch_idx)
            else:
                losses, predictions = self._standard_training_step(batch, batch_idx)
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()

            # Calculate metrics
            metrics = self._calculate_metrics(predictions, batch)
            for key, value in metrics.items():
                epoch_metrics[key] += value

            # Update progress bar (only for main process)
            if self.gpu_wrapper.is_main_process():
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'step': f"{self.training_state.global_step}"
                })
                
                # Log to tensorboard
                if self.writer:
                    self.writer.add_scalar('train/batch_loss', losses['total'].item(), 
                                         self.training_state.global_step)
            
            # Sub-epoch checkpoint saving based on global step
            if self.enhanced_checkpoints and self._should_save_checkpoint_global():
                if self.gpu_wrapper.is_main_process():
                    logger.info(f"Saving checkpoint at global step {self.training_state.global_step}")
                    self.save_checkpoint(is_best=False)

        except Exception as e:
            logger.error(f"Error in training step {batch_idx}: {e}")
            
            # Emergency save if network error
            if self.enhanced_checkpoints and self.network_error_handler:
                error_type, _ = self.network_error_handler._classify_error(e)
                if error_type == ErrorType.NETWORK_ERROR:
                    if self.gpu_wrapper.is_main_process():
                        logger.warning("Network error detected - saving emergency checkpoint")
                        self.save_checkpoint(is_best=False, is_emergency=True)
                    
                # Reraise for handling by parent
                raise
    
    # Update epoch tracking
    if not is_resumed_epoch:
        self.training_state.batches_per_epoch = len(self.train_loader)
    
    # Average metrics
    num_batches = len(self.train_loader)
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    # Store last metrics for checkpoint saving
    self.last_train_metrics = {**epoch_losses, **epoch_metrics}
    
    return epoch_losses, epoch_metrics


# 4. Add new method for global step-based checkpoint saving
def _should_save_checkpoint_global(self):
    """Determine if checkpoint should be saved based on global step"""
    if not self.enhanced_checkpoints:
        return False
    # Save every N global steps
    checkpoint_interval = getattr(self, 'global_checkpoint_freq', 1000)
    return (self.training_state.global_step > 0 and 
            self.training_state.global_step % checkpoint_interval == 0)


# 5. Replace the save_checkpoint method
def save_checkpoint(self, is_best: bool = False, is_emergency: bool = False):
    """Save model checkpoint with complete training state"""
    if not self.gpu_wrapper.is_main_process():
        return  # Only main process saves checkpoints
        
    if self.enhanced_checkpoints and self.checkpoint_manager:
        try:
            # Get current training metrics
            current_metrics = {}
            if hasattr(self, 'last_train_metrics'):
                current_metrics.update(self.last_train_metrics)
            if hasattr(self, 'last_val_metrics'):
                current_metrics.update({f'val_{k}': v for k, v in self.last_val_metrics.items()})
            
            # Create enhanced checkpoint with complete state
            checkpoint = create_resume_compatible_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                training_state=self.training_state,
                scaler=self.scaler if self.use_mixed_precision else None,
                additional_data={
                    'config': self.config,
                    'metrics': current_metrics,
                    'model_config': self.config,
                    'is_emergency': is_emergency,
                    'checkpoint_reason': 'emergency' if is_emergency else ('best' if is_best else 'periodic')
                }
            )
            
            # Save using checkpoint manager for proper tracking
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model_state=checkpoint['model_state_dict'],
                optimizer_state=checkpoint['optimizer_state_dict'],
                scheduler_state=checkpoint['scheduler_state_dict'],
                epoch=self.training_state.epoch,
                step=self.training_state.global_step,  # Use global step
                metrics=current_metrics,
                model_config=self.config,
                is_best=is_best,
                is_emergency=is_emergency,
                blocking=is_emergency  # Emergency saves are always synchronous
            )
            
            if checkpoint_path:
                logger.info(f"🔄 Checkpoint saved: {checkpoint_path} (global step: {self.training_state.global_step})")
                
        except Exception as e:
            logger.error(f"Enhanced checkpoint save failed: {e}")
            # Fallback to basic checkpoint save
            self._fallback_checkpoint_save(is_best, is_emergency)
    else:
        # Use standard checkpoint saving (backward compatible)
        self._fallback_checkpoint_save(is_best, is_emergency)


# 6. Modify the _load_specific_checkpoint method
def _load_specific_checkpoint(self, checkpoint_path: str):
    """Load a specific checkpoint file with enhanced state restoration"""
    try:
        if self.enhanced_checkpoints:
            # Use enhanced loading with compatibility
            success, warnings = load_checkpoint_with_compatibility(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                training_state=self.training_state,
                device=self.device,
                scaler=self.scaler if self.use_mixed_precision else None,
                strict=False
            )
            
            if success:
                # Update instance attributes from training state
                self.epoch = self.training_state.epoch
                self.batch_step = self.training_state.global_step
                self.best_val_loss = self.training_state.best_val_loss
                
                logger.info(f"Resumed from {checkpoint_path}:")
                logger.info(f"  Epoch: {self.epoch}")
                logger.info(f"  Global step: {self.training_state.global_step}")
                logger.info(f"  Epoch step: {self.training_state.epoch_step}")
                logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
                
                if warnings:
                    for warning in warnings:
                        logger.warning(f"  Warning: {warning}")
            else:
                raise Exception(f"Failed to load checkpoint: {warnings}")
                
        else:
            # Fallback to original loading logic
            checkpoint_data, metadata = self.checkpoint_manager.load_checkpoint(
                checkpoint_path, 
                validate=self.checkpoint_validation
            )
            
            # Load model state
            self.model.load_state_dict(checkpoint_data['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Restore training state
            self.epoch = checkpoint_data['epoch']
            self.batch_step = checkpoint_data.get('step', 0) or 0
            self.best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            
            logger.info(f"Resumed from {checkpoint_path}: epoch {self.epoch}, step {self.batch_step}")
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        raise


# 7. Update the train method
def train(self):
    """Main training loop with enhanced resume support"""
    # Update total epochs in training state
    self.training_state.total_epochs = self.num_epochs
    
    # Clear resume info if starting new epoch
    if hasattr(self.training_state, 'resume_info'):
        resumed_epoch = self.training_state.resume_info.get('resumed_epoch', -1)
        if self.epoch > resumed_epoch:
            self.training_state.resume_info = {}
    
    logger.info(f"Starting training from epoch {self.epoch+1}/{self.num_epochs}")
    logger.info(f"Global step: {self.training_state.global_step}")
    logger.info(f"Enhanced checkpoints: {self.enhanced_checkpoints}")
    
    # Create tensorboard writer
    if self.gpu_wrapper.is_main_process():
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
    
    for epoch in range(self.epoch, self.num_epochs):
        self.epoch = epoch
        self.training_state.epoch = epoch
        
        # Training epoch
        if self.enhanced_checkpoints:
            train_losses, train_metrics = self._train_epoch_enhanced()
        else:
            train_losses, train_metrics = self._standard_train_epoch()
        
        # ... rest of training loop ...
