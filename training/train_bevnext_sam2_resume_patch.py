#!/usr/bin/env python3
"""
Patch for train_bevnext_sam2.py to fix resume functionality

This module shows the modifications needed to properly implement resume
functionality in the training script.

Author: Senior Python Programmer & AI Training Expert
"""

# The following modifications should be applied to train_bevnext_sam2.py:

# 1. Import the fixes module
from training.resume_fixes import (
    EnhancedTrainingState, 
    create_resume_compatible_checkpoint,
    load_checkpoint_with_compatibility,
    ResumeAwareDataLoader
)

# 2. Replace the Trainer class __init__ method to include EnhancedTrainingState
def __init__(self, config, use_mixed_precision=True, gpu_ids=None, 
             distributed=False, gradient_accumulation=1, lr_scaling=False):
    """Initialize trainer with configuration"""
    # ... existing initialization code ...
    
    # Replace these lines:
    # self.epoch = 0
    # self.best_val_loss = float('inf')
    # self.batch_step = 0
    # self.training_stats = defaultdict(list)
    
    # With:
    self.training_state = EnhancedTrainingState()
    self.epoch = 0  # Keep for backward compatibility
    self.best_val_loss = float('inf')  # Keep for backward compatibility
    
    # ... rest of initialization ...


# 3. Modify the train_epoch method to properly track steps
def train_epoch(self) -> Dict:
    """Train for one epoch with multi-GPU support and proper step tracking"""
    self.model.train()
    total_loss = 0
    loss_components = {}
    
    # Reset gradient accumulator
    self.grad_accumulator.reset()
    
    # Create resume-aware dataloader
    resume_aware_loader = ResumeAwareDataLoader(self.train_loader, self.training_state)
    
    # Only show progress bar on main process
    if self.gpu_wrapper.is_main_process():
        # Account for skipped batches in progress bar
        initial = self.training_state.epoch_step if self.training_state.resume_info.get('resumed_epoch') == self.epoch else 0
        pbar = tqdm(resume_aware_loader, desc=f'Epoch {self.epoch+1} Train', initial=initial, total=len(self.train_loader))
    else:
        pbar = resume_aware_loader
    
    # Track if this is a resumed epoch
    is_resumed_epoch = self.training_state.resume_info.get('resumed_epoch') == self.epoch
    
    for batch_idx, batch in pbar:
        # Update training state
        self.training_state.update_step(batch_idx, self.epoch)
        
        # ... existing training code ...
        
        # Log batch metrics (add after loss computation)
        if self.gpu_wrapper.is_main_process() and self.writer:
            self.writer.add_scalar('train/batch_loss', losses['total'].item(), self.training_state.global_step)
            
        # Save checkpoint at regular intervals
        if self.training_state.global_step > 0 and self.training_state.global_step % 1000 == 0:
            if self.gpu_wrapper.is_main_process():
                self.logger.info(f"Saving checkpoint at step {self.training_state.global_step}")
                self.save_checkpoint(is_best=False)
    
    # Update epoch tracking
    if not is_resumed_epoch:
        self.training_state.batches_per_epoch = len(self.train_loader)
    
    # ... rest of the method ...


# 4. Replace the save_checkpoint method
def save_checkpoint(self, is_best: bool = False):
    """Save model checkpoint with complete training state"""
    try:
        # Create checkpoint with all necessary state
        checkpoint = create_resume_compatible_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training_state=self.training_state,
            scaler=getattr(self, 'scaler', None),
            additional_data={
                'config': self.config,
                'gpu_wrapper_info': {
                    'num_gpus': self.gpu_wrapper.num_gpus,
                    'distributed': self.gpu_wrapper.distributed
                }
            }
        )
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path} (step {self.training_state.global_step})")
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
            
        # Save periodic checkpoints
        if (self.epoch + 1) % 10 == 0:
            epoch_checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.epoch+1}.pth'
            torch.save(checkpoint, epoch_checkpoint_path)
            self.logger.info(f"Periodic checkpoint saved: {epoch_checkpoint_path}")
            
    except Exception as e:
        self.logger.error(f"Failed to save checkpoint: {e}")
        # Fallback to basic save
        self._fallback_checkpoint_save(is_best)


# 5. Modify the train method to update training state
def train(self):
    """Main training loop with enhanced resume support"""
    # Update total epochs
    self.training_state.total_epochs = self.num_epochs
    
    # Clear resume info if starting new epoch
    if hasattr(self.training_state, 'resume_info'):
        resumed_epoch = self.training_state.resume_info.get('resumed_epoch', -1)
        if self.epoch > resumed_epoch:
            self.training_state.resume_info = {}
    
    self.logger.info(f"Starting training from epoch {self.epoch+1}/{self.num_epochs}")
    self.logger.info(f"Global step: {self.training_state.global_step}")
    
    # ... existing training loop ...
    
    # Update epoch from training state
    for epoch in range(self.epoch, self.num_epochs):
        self.epoch = epoch
        self.training_state.epoch = epoch
        
        # ... rest of training loop ...


# 6. In the main() function, modify the resume logic:
def handle_resume(trainer, args):
    """Handle resume with fixed functionality"""
    if args.auto_resume and not args.resume:
        from training.resume_fixes import create_resume_manager_with_fixes
        
        print("🔍 Searching for resumable checkpoints...")
        # Create fixed resume manager
        resume_manager = create_resume_manager_with_fixes(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            device=trainer.device,
            output_dir=trainer.output_dir,
            scaler=getattr(trainer, 'scaler', None),
            config=trainer.config,
            strict_config_check=False,
            allow_partial_load=True
        )
        
        # Attempt auto-resume
        result = resume_manager.auto_resume()
        
        if result.success:
            print(f"✅ Auto-resume successful!")
            print(f"   📁 Checkpoint: {result.checkpoint_path.name}")
            print(f"   📊 Restored to epoch: {result.restored_epoch}")
            print(f"   📈 Global step: {result.restored_step}")
            
            # Update trainer state from resume manager
            trainer.training_state = resume_manager.training_state
            trainer.epoch = resume_manager.training_state.epoch
            trainer.best_val_loss = resume_manager.training_state.best_val_loss
            
            if result.warnings:
                print(f"   ⚠️  Auto-resume completed with warnings:")
                for warning in result.warnings:
                    print(f"      - {warning}")
        else:
            print("🆕 Starting fresh training...")
            
    elif args.resume:
        # Similar logic for manual resume
        success, warnings = load_checkpoint_with_compatibility(
            checkpoint_path=args.resume,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            training_state=trainer.training_state,
            device=trainer.device,
            scaler=getattr(trainer, 'scaler', None)
        )
        
        if success:
            print(f"✅ Resume successful from: {args.resume}")
            print(f"   Restored to epoch: {trainer.training_state.epoch}")
            print(f"   Global step: {trainer.training_state.global_step}")
            
            # Update trainer attributes
            trainer.epoch = trainer.training_state.epoch
            trainer.best_val_loss = trainer.training_state.best_val_loss
            
            if warnings:
                print(f"   ⚠️  Resume completed with warnings:")
                for warning in warnings:
                    print(f"      - {warning}")
        else:
            print(f"❌ Resume failed: {warnings}")
            print("🔄 Starting fresh training instead...")
