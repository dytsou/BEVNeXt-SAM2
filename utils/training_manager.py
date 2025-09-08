#!/usr/bin/env python3
"""
Training Management Utility for BEVNeXt-SAM2

This utility provides comprehensive training management including:
- Resume operations
- Checkpoint management
- Training monitoring
- Progress tracking

Author: Senior Python Programmer & AI Training Expert
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import signal

# Add training directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "training"))

try:
    from resume_fixes import EnhancedTrainingState
    from checkpoint_manager import CheckpointManager
    from auto_resume import AutoResumeManager
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Warning: Enhanced training features not available")


class TrainingManager:
    """Comprehensive training management system"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.outputs_dir = self.project_root / "outputs"
        self.training_dir = self.outputs_dir / "training"
        self.logs_dir = self.project_root / "logs"
        
        # Ensure directories exist
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize enhanced components if available
        if ENHANCED_AVAILABLE:
            try:
                self.checkpoint_manager = CheckpointManager(
                    checkpoint_dir=self.training_dir,
                    keep_latest=5,
                    keep_best=3,
                    enable_compression=False
                )
                self.auto_resume_manager = AutoResumeManager(
                    checkpoint_manager=self.checkpoint_manager,
                    interactive=False
                )
            except Exception as e:
                print(f"Warning: Could not initialize enhanced managers: {e}")
                self.checkpoint_manager = None
                self.auto_resume_manager = None
        else:
            self.checkpoint_manager = None
            self.auto_resume_manager = None
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        status = {
            'training_active': False,
            'container_running': False,
            'latest_checkpoint': None,
            'best_checkpoint': None,
            'training_progress': None,
            'resume_available': False,
            'disk_usage_mb': 0
        }
        
        # Check if training container is running
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'ancestor=bevnext-sam2:latest', '--format', '{{.ID}}'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                status['training_active'] = True
                status['container_running'] = True
                status['container_id'] = result.stdout.strip().split('\n')[0]
        except:
            pass
        
        # Check for checkpoints
        latest_checkpoint = self.training_dir / "checkpoint_latest.pth"
        if latest_checkpoint.exists():
            status['latest_checkpoint'] = str(latest_checkpoint)
            
            # Try to extract checkpoint info
            try:
                import torch
                ckpt = torch.load(latest_checkpoint, map_location='cpu')
                status['training_progress'] = {
                    'epoch': ckpt.get('epoch', 'unknown'),
                    'global_step': ckpt.get('global_step', ckpt.get('step', ckpt.get('batch_step', 'unknown'))),
                    'timestamp': ckpt.get('timestamp', 'unknown'),
                    'best_val_loss': ckpt.get('best_val_loss', 'unknown')
                }
                status['resume_available'] = True
            except:
                pass
        
        best_checkpoint = self.training_dir / "checkpoint_best.pth"
        if best_checkpoint.exists():
            status['best_checkpoint'] = str(best_checkpoint)
        
        # Calculate disk usage
        try:
            total_size = 0
            for file_path in self.training_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            status['disk_usage_mb'] = total_size / (1024 * 1024)
        except:
            pass
        
        return status
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        if self.checkpoint_manager:
            # Use enhanced checkpoint manager
            available_checkpoints = self.checkpoint_manager.list_available_checkpoints()
            for checkpoint_path, metadata in available_checkpoints:
                checkpoints.append({
                    'path': str(checkpoint_path),
                    'epoch': metadata.epoch,
                    'step': metadata.step,
                    'timestamp': metadata.timestamp,
                    'validation_loss': metadata.validation_loss,
                    'is_best': metadata.is_best,
                    'size_mb': metadata.model_size_mb,
                    'resume_count': metadata.resume_count
                })
        else:
            # Fallback: scan directory for checkpoint files
            for checkpoint_file in self.training_dir.glob("checkpoint_*.pth*"):
                try:
                    import torch
                    ckpt = torch.load(checkpoint_file, map_location='cpu')
                    file_stats = checkpoint_file.stat()
                    
                    checkpoints.append({
                        'path': str(checkpoint_file),
                        'epoch': ckpt.get('epoch', 'unknown'),
                        'step': ckpt.get('global_step', ckpt.get('step', ckpt.get('batch_step', 'unknown'))),
                        'timestamp': ckpt.get('timestamp', 'unknown'),
                        'validation_loss': ckpt.get('best_val_loss', 'unknown'),
                        'is_best': 'best' in checkpoint_file.name,
                        'size_mb': file_stats.st_size / (1024 * 1024),
                        'resume_count': ckpt.get('resume_count', 0)
                    })
                except:
                    # Add basic file info if checkpoint can't be loaded
                    file_stats = checkpoint_file.stat()
                    checkpoints.append({
                        'path': str(checkpoint_file),
                        'epoch': 'unknown',
                        'step': 'unknown',
                        'timestamp': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        'validation_loss': 'unknown',
                        'is_best': 'best' in checkpoint_file.name,
                        'size_mb': file_stats.st_size / (1024 * 1024),
                        'resume_count': 0
                    })
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def detect_resume_opportunity(self) -> Dict[str, Any]:
        """Detect and analyze resume opportunities"""
        result = {
            'can_resume': False,
            'recommended_checkpoint': None,
            'confidence_score': 0.0,
            'resume_reason': 'No suitable checkpoint found'
        }
        
        if self.auto_resume_manager:
            # Use enhanced auto-resume detection
            resume_info = self.auto_resume_manager.detect_resume_opportunity()
            if resume_info:
                result.update({
                    'can_resume': True,
                    'recommended_checkpoint': str(resume_info['checkpoint_path']),
                    'confidence_score': resume_info['confidence_score'],
                    'resume_reason': resume_info['resume_reason'],
                    'epoch': resume_info['epoch'],
                    'step': resume_info['step'],
                    'age_hours': resume_info['age_hours']
                })
        else:
            # Fallback: simple checkpoint detection
            latest_checkpoint = self.training_dir / "checkpoint_latest.pth"
            if latest_checkpoint.exists():
                file_age = time.time() - latest_checkpoint.stat().st_mtime
                age_hours = file_age / 3600
                
                if age_hours < 24:  # Recent enough to resume
                    result.update({
                        'can_resume': True,
                        'recommended_checkpoint': str(latest_checkpoint),
                        'confidence_score': max(0.5, 1.0 - (age_hours / 24)),
                        'resume_reason': f'Latest checkpoint ({age_hours:.1f}h old)',
                        'age_hours': age_hours
                    })
        
        return result
    
    def start_training(self, args: Dict[str, Any]) -> bool:
        """Start training with specified arguments"""
        # Build command
        cmd = ['./scripts/run.sh', 'train']
        
        # Add arguments
        if args.get('data_path'):
            cmd.extend(['--data-path', args['data_path']])
        
        if args.get('gpu'):
            cmd.append('--gpu')
        
        if args.get('auto_resume'):
            cmd.append('--auto-resume')
        
        if args.get('resume_path'):
            cmd.extend(['--resume', args['resume_path']])
        
        if args.get('no_resume'):
            cmd.append('--no-resume')
        
        if args.get('epochs'):
            cmd.extend(['--epochs', str(args['epochs'])])
        
        if args.get('batch_size'):
            cmd.extend(['--batch-size', str(args['batch_size'])])
        
        if args.get('checkpoint_freq'):
            cmd.extend(['--checkpoint-freq', str(args['checkpoint_freq'])])
        
        print(f"🚀 Starting training with command: {' '.join(cmd)}")
        
        try:
            # Start training process
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor training output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            return return_code == 0
            
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            try:
                process.terminate()
                process.wait(timeout=10)
            except:
                process.kill()
            return False
        except Exception as e:
            print(f"❌ Failed to start training: {e}")
            return False
    
    def stop_training(self) -> bool:
        """Stop currently running training"""
        try:
            # Find training container
            result = subprocess.run(
                ['docker', 'ps', '--filter', 'ancestor=bevnext-sam2:latest', '--format', '{{.ID}}'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                container_id = result.stdout.strip().split('\n')[0]
                
                print(f"⏹️  Stopping training container: {container_id}")
                
                # Send graceful stop signal
                subprocess.run(['docker', 'stop', container_id], timeout=30)
                
                print("✅ Training stopped successfully")
                return True
            else:
                print("ℹ️  No training container found")
                return False
                
        except Exception as e:
            print(f"❌ Failed to stop training: {e}")
            return False
    
    def cleanup_checkpoints(self, keep_latest: int = 5, keep_best: int = 3) -> int:
        """Clean up old checkpoints"""
        if self.checkpoint_manager:
            # Use enhanced checkpoint manager
            try:
                self.checkpoint_manager.cleanup_old_checkpoints(force=True)
                print("✅ Checkpoint cleanup completed using enhanced manager")
                return 0
            except Exception as e:
                print(f"Warning: Enhanced cleanup failed: {e}")
        
        # Fallback cleanup
        checkpoints = self.list_checkpoints()
        
        # Separate into categories
        latest_checkpoints = [c for c in checkpoints if not c['is_best']]
        best_checkpoints = [c for c in checkpoints if c['is_best']]
        
        # Sort by timestamp
        latest_checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        best_checkpoints.sort(key=lambda x: x.get('validation_loss', float('inf')))
        
        # Determine which to keep
        keep_latest_paths = {c['path'] for c in latest_checkpoints[:keep_latest]}
        keep_best_paths = {c['path'] for c in best_checkpoints[:keep_best]}
        keep_paths = keep_latest_paths | keep_best_paths
        
        # Remove others
        removed_count = 0
        for checkpoint in checkpoints:
            if checkpoint['path'] not in keep_paths:
                try:
                    Path(checkpoint['path']).unlink()
                    removed_count += 1
                    print(f"🗑️  Removed: {Path(checkpoint['path']).name}")
                except Exception as e:
                    print(f"⚠️  Could not remove {checkpoint['path']}: {e}")
        
        print(f"✅ Cleanup completed: removed {removed_count} checkpoints")
        return removed_count
    
    def get_training_logs(self, lines: int = 50) -> List[str]:
        """Get recent training log entries"""
        log_entries = []
        
        # Check for training log file
        log_file = self.training_dir / "training.log"
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    all_lines = f.readlines()
                    log_entries = all_lines[-lines:] if len(all_lines) > lines else all_lines
            except Exception as e:
                log_entries.append(f"Error reading log file: {e}")
        
        # Also check Docker container logs if training is active
        status = self.get_training_status()
        if status['container_running']:
            try:
                result = subprocess.run(
                    ['docker', 'logs', '--tail', str(lines), status['container_id']],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    docker_logs = result.stdout.split('\n')
                    log_entries.extend(docker_logs)
            except:
                pass
        
        return [line.strip() for line in log_entries if line.strip()]


def main():
    parser = argparse.ArgumentParser(description='BEVNeXt-SAM2 Training Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show training status')
    
    # List checkpoints command
    list_parser = subparsers.add_parser('list', help='List available checkpoints')
    list_parser.add_argument('--detail', action='store_true', help='Show detailed information')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Detect and manage resume opportunities')
    resume_parser.add_argument('--auto', action='store_true', help='Automatically start resume')
    
    # Start training command
    start_parser = subparsers.add_parser('start', help='Start training')
    start_parser.add_argument('--data-path', help='Path to dataset')
    start_parser.add_argument('--gpu', action='store_true', help='Use GPU')
    start_parser.add_argument('--auto-resume', action='store_true', help='Auto-resume from checkpoint')
    start_parser.add_argument('--resume', help='Resume from specific checkpoint')
    start_parser.add_argument('--no-resume', action='store_true', help='Force fresh training')
    start_parser.add_argument('--epochs', type=int, help='Number of epochs')
    start_parser.add_argument('--batch-size', type=int, help='Batch size')
    start_parser.add_argument('--checkpoint-freq', type=int, help='Checkpoint frequency')
    
    # Stop training command
    stop_parser = subparsers.add_parser('stop', help='Stop training')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old checkpoints')
    cleanup_parser.add_argument('--keep-latest', type=int, default=5, help='Number of latest checkpoints to keep')
    cleanup_parser.add_argument('--keep-best', type=int, default=3, help='Number of best checkpoints to keep')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show training logs')
    logs_parser.add_argument('--lines', type=int, default=50, help='Number of log lines to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize manager
    project_root = Path(__file__).parent.parent
    manager = TrainingManager(str(project_root))
    
    # Execute command
    if args.command == 'status':
        status = manager.get_training_status()
        print("📊 TRAINING STATUS")
        print("=" * 50)
        print(f"Training active: {'Yes' if status['training_active'] else 'No'}")
        print(f"Container running: {'Yes' if status['container_running'] else 'No'}")
        print(f"Resume available: {'Yes' if status['resume_available'] else 'No'}")
        print(f"Disk usage: {status['disk_usage_mb']:.1f} MB")
        
        if status['training_progress']:
            prog = status['training_progress']
            print(f"Latest checkpoint:")
            print(f"  Epoch: {prog['epoch']}")
            print(f"  Step: {prog['global_step']}")
            print(f"  Best val loss: {prog['best_val_loss']}")
    
    elif args.command == 'list':
        checkpoints = manager.list_checkpoints()
        print(f"📋 AVAILABLE CHECKPOINTS ({len(checkpoints)} found)")
        print("=" * 80)
        
        if args.detail:
            for ckpt in checkpoints:
                print(f"File: {Path(ckpt['path']).name}")
                print(f"  Epoch: {ckpt['epoch']}")
                print(f"  Step: {ckpt['step']}")
                print(f"  Size: {ckpt['size_mb']:.1f} MB")
                print(f"  Best: {ckpt['is_best']}")
                print(f"  Resume count: {ckpt['resume_count']}")
                print()
        else:
            print(f"{'Filename':<30} {'Epoch':<8} {'Step':<12} {'Size (MB)':<10} {'Best':<6}")
            print("-" * 80)
            for ckpt in checkpoints:
                filename = Path(ckpt['path']).name
                print(f"{filename:<30} {ckpt['epoch']:<8} {ckpt['step']:<12} "
                      f"{ckpt['size_mb']:<10.1f} {'Yes' if ckpt['is_best'] else 'No':<6}")
    
    elif args.command == 'resume':
        resume_info = manager.detect_resume_opportunity()
        print("🔄 RESUME ANALYSIS")
        print("=" * 50)
        print(f"Can resume: {'Yes' if resume_info['can_resume'] else 'No'}")
        
        if resume_info['can_resume']:
            print(f"Recommended checkpoint: {Path(resume_info['recommended_checkpoint']).name}")
            print(f"Confidence score: {resume_info['confidence_score']:.2f}")
            print(f"Reason: {resume_info['resume_reason']}")
            
            if args.auto:
                print("\n🚀 Starting auto-resume...")
                training_args = {
                    'auto_resume': True,
                    'gpu': True
                }
                success = manager.start_training(training_args)
                return 0 if success else 1
        else:
            print(f"Reason: {resume_info['resume_reason']}")
    
    elif args.command == 'start':
        training_args = vars(args)
        training_args.pop('command')
        success = manager.start_training(training_args)
        return 0 if success else 1
    
    elif args.command == 'stop':
        success = manager.stop_training()
        return 0 if success else 1
    
    elif args.command == 'cleanup':
        removed = manager.cleanup_checkpoints(args.keep_latest, args.keep_best)
        return 0
    
    elif args.command == 'logs':
        logs = manager.get_training_logs(args.lines)
        print(f"📄 TRAINING LOGS (last {len(logs)} lines)")
        print("=" * 50)
        for line in logs:
            print(line)
    
    return 0


if __name__ == '__main__':
    exit(main())
