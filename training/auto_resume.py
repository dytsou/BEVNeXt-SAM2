#!/usr/bin/env python3
"""
Auto-Resume Detection System for Training Continuity

This module provides intelligent checkpoint detection and selection for
automatic training resumption after interruptions.

Author: Senior Python Programmer & AI Training Expert
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

from .checkpoint_manager import CheckpointManager, CheckpointMetadata

logger = logging.getLogger(__name__)


@dataclass
class ResumeCandidate:
    """Information about a potential resume checkpoint"""
    checkpoint_path: Path
    metadata: CheckpointMetadata
    score: float
    reason: str
    is_valid: bool
    age_hours: float
    
    def __str__(self) -> str:
        return (f"ResumeCandidate(path={self.checkpoint_path.name}, "
                f"epoch={self.metadata.epoch}, step={self.metadata.step}, "
                f"score={self.score:.2f}, reason='{self.reason}')")


class AutoResumeManager:
    """Intelligent auto-resume detection and management"""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        interactive: bool = True,
        auto_resume_threshold_hours: float = 24.0,
        min_epochs_for_auto_resume: int = 1,
        enable_emergency_resume: bool = True
    ):
        """
        Initialize auto-resume manager
        
        Args:
            checkpoint_manager: Checkpoint manager instance
            interactive: Enable interactive prompts for user confirmation
            auto_resume_threshold_hours: Auto-resume if checkpoint is newer than this
            min_epochs_for_auto_resume: Minimum epochs required for auto-resume
            enable_emergency_resume: Enable resume from emergency checkpoints
        """
        self.checkpoint_manager = checkpoint_manager
        self.interactive = interactive
        self.auto_resume_threshold_hours = auto_resume_threshold_hours
        self.min_epochs_for_auto_resume = min_epochs_for_auto_resume
        self.enable_emergency_resume = enable_emergency_resume
        
        logger.info(f"AutoResumeManager initialized (interactive={interactive})")

    def detect_resume_opportunity(self) -> Optional[Dict[str, Any]]:
        """
        Detect if training can be resumed and return resume information
        
        Returns:
            Resume information dict or None if no resume opportunity
        """
        logger.info("Detecting training resume opportunities...")
        
        # Find all resume candidates
        candidates = self._find_resume_candidates()
        
        if not candidates:
            logger.info("No valid resume candidates found")
            return None
        
        # Log all candidates
        logger.info(f"Found {len(candidates)} resume candidates:")
        for i, candidate in enumerate(candidates, 1):
            logger.info(f"  {i}. {candidate}")
        
        # Select best candidate
        best_candidate = self._select_best_candidate(candidates)
        
        if not best_candidate:
            logger.info("No suitable resume candidate selected")
            return None
        
        # Check if auto-resume is appropriate
        if self._should_auto_resume(best_candidate):
            logger.info(f"Auto-resuming from: {best_candidate.checkpoint_path}")
            return self._create_resume_info(best_candidate, auto_resumed=True)
        
        # Interactive confirmation if enabled
        if self.interactive:
            if self._prompt_user_confirmation(best_candidate):
                logger.info(f"User confirmed resume from: {best_candidate.checkpoint_path}")
                return self._create_resume_info(best_candidate, auto_resumed=False)
            else:
                logger.info("User declined resume opportunity")
                return None
        
        # Non-interactive mode - use best candidate if meets criteria
        if best_candidate.score >= 0.7:  # High confidence threshold
            logger.info(f"Non-interactive resume from: {best_candidate.checkpoint_path}")
            return self._create_resume_info(best_candidate, auto_resumed=True)
        
        logger.info("Resume candidate score too low for non-interactive mode")
        return None

    def _find_resume_candidates(self) -> List[ResumeCandidate]:
        """Find all potential resume candidates"""
        candidates = []
        
        # Get all available checkpoints
        available_checkpoints = self.checkpoint_manager.list_available_checkpoints()
        
        for checkpoint_path, metadata in available_checkpoints:
            # Skip if too few epochs for meaningful resume
            if metadata.epoch < self.min_epochs_for_auto_resume:
                continue
            
            # Calculate age
            age_hours = (time.time() - metadata.creation_time) / 3600
            
            # Score the candidate
            score, reason = self._score_resume_candidate(metadata, age_hours)
            
            # Validate checkpoint
            is_valid = self.checkpoint_manager.validate_checkpoint(checkpoint_path)
            
            candidate = ResumeCandidate(
                checkpoint_path=checkpoint_path,
                metadata=metadata,
                score=score,
                reason=reason,
                is_valid=is_valid,
                age_hours=age_hours
            )
            
            if is_valid:
                candidates.append(candidate)
            else:
                logger.warning(f"Invalid checkpoint skipped: {checkpoint_path}")
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates

    def _score_resume_candidate(self, metadata: CheckpointMetadata, age_hours: float) -> Tuple[float, str]:
        """
        Score a resume candidate based on various factors
        
        Returns:
            Tuple of (score, reason)
        """
        score = 0.0
        reasons = []
        
        # Age factor (newer is better, but not too new to avoid corruption)
        if age_hours < 0.1:  # Less than 6 minutes - might be corrupted
            age_score = 0.3
            reasons.append("very recent")
        elif age_hours < 1.0:  # Less than 1 hour - good
            age_score = 0.9
            reasons.append("recent")
        elif age_hours < 6.0:  # Less than 6 hours - acceptable
            age_score = 0.8
            reasons.append("moderately recent")
        elif age_hours < 24.0:  # Less than 24 hours - ok
            age_score = 0.6
            reasons.append("within day")
        else:  # Older than 24 hours - questionable
            age_score = 0.3
            reasons.append("old")
        
        score += age_score * 0.3
        
        # Progress factor (more epochs is better)
        if metadata.epoch >= 50:
            progress_score = 1.0
            reasons.append("substantial progress")
        elif metadata.epoch >= 20:
            progress_score = 0.8
            reasons.append("good progress")
        elif metadata.epoch >= 10:
            progress_score = 0.6
            reasons.append("some progress")
        elif metadata.epoch >= 5:
            progress_score = 0.4
            reasons.append("early progress")
        else:
            progress_score = 0.2
            reasons.append("minimal progress")
        
        score += progress_score * 0.3
        
        # Validation loss factor (lower is better)
        if metadata.validation_loss is not None:
            if metadata.validation_loss < 0.1:
                val_score = 1.0
                reasons.append("excellent validation")
            elif metadata.validation_loss < 0.5:
                val_score = 0.8
                reasons.append("good validation")
            elif metadata.validation_loss < 1.0:
                val_score = 0.6
                reasons.append("acceptable validation")
            else:
                val_score = 0.4
                reasons.append("poor validation")
        else:
            val_score = 0.5
            reasons.append("no validation data")
        
        score += val_score * 0.2
        
        # Step factor (sub-epoch progress)
        if metadata.step is not None and metadata.step > 0:
            step_score = 0.8
            reasons.append("sub-epoch progress")
        else:
            step_score = 0.5
            reasons.append("epoch boundary")
        
        score += step_score * 0.1
        
        # Resume count factor (fewer resumes is better)
        if metadata.resume_count == 0:
            resume_score = 1.0
            reasons.append("never resumed")
        elif metadata.resume_count <= 2:
            resume_score = 0.8
            reasons.append("rarely resumed")
        elif metadata.resume_count <= 5:
            resume_score = 0.6
            reasons.append("sometimes resumed")
        else:
            resume_score = 0.4
            reasons.append("frequently resumed")
        
        score += resume_score * 0.1
        
        # Ensure score is in [0, 1]
        score = max(0.0, min(1.0, score))
        
        reason = ", ".join(reasons)
        
        return score, reason

    def _select_best_candidate(self, candidates: List[ResumeCandidate]) -> Optional[ResumeCandidate]:
        """Select the best resume candidate"""
        if not candidates:
            return None
        
        # Filter valid candidates
        valid_candidates = [c for c in candidates if c.is_valid]
        
        if not valid_candidates:
            logger.warning("No valid candidates found")
            return None
        
        # Return highest scoring candidate
        best = valid_candidates[0]
        logger.info(f"Best candidate: {best}")
        
        return best

    def _should_auto_resume(self, candidate: ResumeCandidate) -> bool:
        """Determine if we should automatically resume without user confirmation"""
        # Auto-resume criteria
        criteria = [
            candidate.score >= 0.8,  # High confidence score
            candidate.age_hours <= self.auto_resume_threshold_hours,  # Recent enough
            candidate.metadata.epoch >= self.min_epochs_for_auto_resume,  # Meaningful progress
            candidate.metadata.resume_count <= 3,  # Not resumed too many times
        ]
        
        auto_resume = all(criteria)
        
        if auto_resume:
            logger.info("Auto-resume criteria met:")
            logger.info(f"  Score: {candidate.score:.2f} >= 0.8")
            logger.info(f"  Age: {candidate.age_hours:.1f}h <= {self.auto_resume_threshold_hours}h")
            logger.info(f"  Epochs: {candidate.metadata.epoch} >= {self.min_epochs_for_auto_resume}")
            logger.info(f"  Resume count: {candidate.metadata.resume_count} <= 3")
        
        return auto_resume

    def _prompt_user_confirmation(self, candidate: ResumeCandidate) -> bool:
        """Prompt user for resume confirmation"""
        print("\n" + "="*60)
        print("🔄 TRAINING RESUME OPPORTUNITY DETECTED")
        print("="*60)
        
        print(f"📁 Checkpoint: {candidate.checkpoint_path.name}")
        print(f"📊 Progress: Epoch {candidate.metadata.epoch}", end="")
        if candidate.metadata.step:
            print(f", Step {candidate.metadata.step}")
        else:
            print()
        
        print(f"⏰ Created: {datetime.fromtimestamp(candidate.metadata.creation_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 Age: {candidate.age_hours:.1f} hours ago")
        
        if candidate.metadata.validation_loss:
            print(f"📈 Validation Loss: {candidate.metadata.validation_loss:.4f}")
        
        print(f"⭐ Confidence Score: {candidate.score:.2f}/1.0 ({candidate.reason})")
        print(f"🔄 Previous Resumes: {candidate.metadata.resume_count}")
        
        print("\n" + "-"*60)
        
        while True:
            try:
                response = input("Resume training from this checkpoint? [Y/n/info]: ").strip().lower()
                
                if response in ['', 'y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                elif response in ['i', 'info']:
                    self._show_detailed_info(candidate)
                else:
                    print("Please enter 'y' for yes, 'n' for no, or 'info' for details")
                    
            except (KeyboardInterrupt, EOFError):
                print("\nUser interrupted - continuing with fresh training")
                return False

    def _show_detailed_info(self, candidate: ResumeCandidate):
        """Show detailed information about the candidate"""
        print("\n" + "="*50)
        print("📋 DETAILED CHECKPOINT INFORMATION")
        print("="*50)
        
        metadata = candidate.metadata
        
        print(f"📁 File: {candidate.checkpoint_path}")
        print(f"📊 Epoch: {metadata.epoch}")
        print(f"🔢 Step: {metadata.step or 'N/A'}")
        print(f"📅 Timestamp: {metadata.timestamp}")
        print(f"⚖️ Model Size: {metadata.model_size_mb:.1f} MB")
        print(f"🧮 Parameters: {metadata.total_params:,}")
        print(f"🔄 Resume Count: {metadata.resume_count}")
        print(f"✅ Is Best: {metadata.is_best}")
        print(f"🗜️ Compressed: {metadata.compressed}")
        
        if metadata.training_metrics:
            print(f"📈 Training Metrics:")
            for key, value in metadata.training_metrics.items():
                print(f"   {key}: {value}")
        
        if metadata.model_config:
            print(f"⚙️ Model Config:")
            for key, value in metadata.model_config.items():
                print(f"   {key}: {value}")
        
        print("-"*50)

    def _create_resume_info(self, candidate: ResumeCandidate, auto_resumed: bool) -> Dict[str, Any]:
        """Create resume information dictionary"""
        return {
            'checkpoint_path': candidate.checkpoint_path,
            'metadata': candidate.metadata,
            'auto_resumed': auto_resumed,
            'resume_reason': candidate.reason,
            'confidence_score': candidate.score,
            'age_hours': candidate.age_hours,
            'epoch': candidate.metadata.epoch,
            'step': candidate.metadata.step,
            'resume_timestamp': datetime.now().isoformat()
        }

    def select_checkpoint_for_resume(
        self, 
        preference: str = 'latest'
    ) -> Optional[Tuple[Path, CheckpointMetadata]]:
        """
        Select specific checkpoint for resume based on preference
        
        Args:
            preference: 'latest', 'best', or specific epoch number
            
        Returns:
            Tuple of (checkpoint_path, metadata) or None
        """
        available_checkpoints = self.checkpoint_manager.list_available_checkpoints()
        
        if not available_checkpoints:
            return None
        
        if preference == 'latest':
            # Return most recent checkpoint
            return available_checkpoints[0]
        
        elif preference == 'best':
            # Return checkpoint with best validation loss
            best_checkpoint = None
            best_loss = float('inf')
            
            for checkpoint_path, metadata in available_checkpoints:
                if metadata.validation_loss is not None and metadata.validation_loss < best_loss:
                    best_checkpoint = (checkpoint_path, metadata)
                    best_loss = metadata.validation_loss
            
            return best_checkpoint
        
        else:
            # Try to parse as epoch number
            try:
                target_epoch = int(preference)
                for checkpoint_path, metadata in available_checkpoints:
                    if metadata.epoch == target_epoch:
                        return (checkpoint_path, metadata)
            except ValueError:
                logger.warning(f"Invalid preference: {preference}")
        
        return None

    def get_resume_statistics(self) -> Dict[str, Any]:
        """Get statistics about available checkpoints for resume"""
        available_checkpoints = self.checkpoint_manager.list_available_checkpoints()
        
        if not available_checkpoints:
            return {'total_checkpoints': 0}
        
        epochs = [metadata.epoch for _, metadata in available_checkpoints]
        val_losses = [metadata.validation_loss for _, metadata in available_checkpoints 
                     if metadata.validation_loss is not None]
        ages_hours = [(time.time() - metadata.creation_time) / 3600 
                     for _, metadata in available_checkpoints]
        
        stats = {
            'total_checkpoints': len(available_checkpoints),
            'epoch_range': (min(epochs), max(epochs)),
            'latest_epoch': max(epochs),
            'oldest_checkpoint_hours': max(ages_hours),
            'newest_checkpoint_hours': min(ages_hours),
            'has_validation_data': len(val_losses) > 0,
        }
        
        if val_losses:
            stats['best_validation_loss'] = min(val_losses)
            stats['latest_validation_loss'] = val_losses[0]  # First is most recent
        
        return stats

    def cleanup_invalid_checkpoints(self) -> int:
        """Remove invalid checkpoints and return count of removed files"""
        available_checkpoints = self.checkpoint_manager.list_available_checkpoints()
        removed_count = 0
        
        for checkpoint_path, metadata in available_checkpoints:
            if not self.checkpoint_manager.validate_checkpoint(checkpoint_path):
                try:
                    checkpoint_path.unlink()
                    removed_count += 1
                    logger.info(f"Removed invalid checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove invalid checkpoint {checkpoint_path}: {e}")
        
        return removed_count

    def force_resume_from_emergency(self) -> Optional[Dict[str, Any]]:
        """Force resume from emergency checkpoint if available"""
        if not self.enable_emergency_resume:
            return None
        
        # Look for emergency checkpoints
        checkpoint_dir = self.checkpoint_manager.checkpoint_dir
        emergency_files = list(checkpoint_dir.glob('emergency_checkpoint_*.pth*'))
        
        if not emergency_files:
            return None
        
        # Sort by modification time (most recent first)
        emergency_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for emergency_file in emergency_files:
            if self.checkpoint_manager.validate_checkpoint(emergency_file):
                logger.warning(f"Emergency resume from: {emergency_file}")
                
                # Load checkpoint to get metadata
                try:
                    checkpoint_data, _ = self.checkpoint_manager.load_checkpoint(
                        emergency_file, validate=False
                    )
                    
                    # Create fake metadata for emergency checkpoint
                    fake_metadata = CheckpointMetadata(
                        checkpoint_path=str(emergency_file),
                        epoch=checkpoint_data.get('epoch', 0),
                        step=checkpoint_data.get('step'),
                        timestamp=checkpoint_data.get('timestamp', datetime.now().isoformat()),
                        model_config=checkpoint_data.get('model_config', {}),
                        training_metrics=checkpoint_data.get('metrics', {}),
                        optimizer_type='unknown',
                        scheduler_type='unknown',
                        total_params=0,
                        model_size_mb=0.0,
                        validation_loss=None,
                        is_best=False,
                        resume_count=0,
                        creation_time=emergency_file.stat().st_mtime,
                        checksum='',
                        compressed=emergency_file.suffix == '.gz'
                    )
                    
                    return {
                        'checkpoint_path': emergency_file,
                        'metadata': fake_metadata,
                        'auto_resumed': True,
                        'resume_reason': 'emergency checkpoint',
                        'confidence_score': 0.5,
                        'age_hours': (time.time() - emergency_file.stat().st_mtime) / 3600,
                        'epoch': fake_metadata.epoch,
                        'step': fake_metadata.step,
                        'resume_timestamp': datetime.now().isoformat(),
                        'is_emergency': True
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to load emergency checkpoint {emergency_file}: {e}")
        
        logger.warning("No valid emergency checkpoints found")
        return None


def create_auto_resume_manager(
    checkpoint_manager: CheckpointManager,
    **kwargs
) -> AutoResumeManager:
    """Factory function to create auto-resume manager"""
    return AutoResumeManager(checkpoint_manager, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    import tempfile
    from checkpoint_manager import create_checkpoint_manager
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create managers
        checkpoint_manager = create_checkpoint_manager(temp_dir)
        auto_resume_manager = create_auto_resume_manager(
            checkpoint_manager, 
            interactive=False
        )
        
        # Check for resume opportunities (should be none)
        resume_info = auto_resume_manager.detect_resume_opportunity()
        print(f"Resume opportunity: {resume_info}")
        
        # Get statistics
        stats = auto_resume_manager.get_resume_statistics()
        print(f"Resume statistics: {stats}")
