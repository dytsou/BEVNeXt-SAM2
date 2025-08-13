#!/usr/bin/env python3
"""
Multi-GPU Training Utilities for BEVNeXt-SAM2
Provides support for both DataParallel and DistributedDataParallel training
"""

import os
import math
import logging
from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class MultiGPUWrapper:
    """
    Wrapper class for multi-GPU training support
    
    Supports both DataParallel and DistributedDataParallel modes
    with automatic GPU detection and configuration
    """
    
    def __init__(
        self, 
        gpu_ids: Optional[Union[List[int], str]] = None,
        distributed: bool = False,
        backend: str = 'nccl'
    ):
        """
        Initialize Multi-GPU wrapper
        
        Args:
            gpu_ids: List of GPU IDs or comma-separated string (e.g., "0,1")
            distributed: Whether to use DistributedDataParallel
            backend: Distributed backend ('nccl' for NVIDIA GPUs)
        """
        self.distributed = distributed
        self.backend = backend
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        
        # Parse GPU IDs
        if gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, str):
            self.gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
        else:
            self.gpu_ids = gpu_ids
            
        # Validate GPU availability
        available_gpus = torch.cuda.device_count()
        if not self.gpu_ids:
            raise ValueError("No GPU IDs specified and no CUDA devices available")
            
        for gpu_id in self.gpu_ids:
            if gpu_id >= available_gpus:
                raise ValueError(f"GPU ID {gpu_id} not available. Only {available_gpus} GPUs found")
        
        self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
        self.num_gpus = len(self.gpu_ids)
        
        logger.info(f"MultiGPUWrapper initialized:")
        logger.info(f"  - Mode: {'DistributedDataParallel' if distributed else 'DataParallel'}")
        logger.info(f"  - GPUs: {self.gpu_ids}")
        logger.info(f"  - Primary device: {self.device}")
        
    def setup_distributed(self) -> Tuple[int, int, int]:
        """
        Setup distributed training environment
        
        Returns:
            Tuple of (gpu, rank, world_size)
        """
        if not self.distributed:
            return self.gpu_ids[0], 0, 1
            
        # Get distributed training parameters from environment
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            # Single machine multi-GPU setup
            self.rank = 0
            self.world_size = self.num_gpus
            self.local_rank = 0
            
        # Set the GPU for this process
        gpu = self.gpu_ids[self.local_rank] if self.local_rank < len(self.gpu_ids) else self.gpu_ids[0]
        torch.cuda.set_device(gpu)
        self.device = torch.device(f'cuda:{gpu}')
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
            
        logger.info(f"Distributed training setup:")
        logger.info(f"  - Rank: {self.rank}")
        logger.info(f"  - World size: {self.world_size}")
        logger.info(f"  - Local rank: {self.local_rank}")
        logger.info(f"  - Device: {self.device}")
        
        return gpu, self.rank, self.world_size
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Wrap model for multi-GPU training
        
        Args:
            model: PyTorch model to wrap
            
        Returns:
            Wrapped model ready for multi-GPU training
        """
        # Move model to primary device first
        model = model.to(self.device)
        
        if self.num_gpus == 1:
            logger.info("Single GPU mode - no wrapping needed")
            return model
        
        if self.distributed:
            # DistributedDataParallel mode
            if not dist.is_initialized():
                self.setup_distributed()
            
            model = DDP(
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=False  # Set to True if you have unused parameters
            )
            logger.info(f"Model wrapped with DistributedDataParallel on device {self.device}")
        else:
            # DataParallel mode
            model = DataParallel(model, device_ids=self.gpu_ids)
            logger.info(f"Model wrapped with DataParallel on devices {self.gpu_ids}")
            
        return model
    
    def create_distributed_sampler(self, dataset) -> Optional[DistributedSampler]:
        """
        Create distributed sampler for dataset
        
        Args:
            dataset: Dataset to create sampler for
            
        Returns:
            DistributedSampler if distributed mode, None otherwise
        """
        if self.distributed and self.num_gpus > 1:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        return None
    
    def scale_batch_size(self, base_batch_size: int, per_gpu: bool = True) -> int:
        """
        Scale batch size based on number of GPUs
        
        Args:
            base_batch_size: Base batch size
            per_gpu: If True, return per-GPU batch size; if False, return total batch size
            
        Returns:
            Scaled batch size
        """
        if per_gpu:
            return base_batch_size
        else:
            return base_batch_size * self.num_gpus
    
    def scale_learning_rate(
        self, 
        base_lr: float, 
        scaling_rule: str = 'linear',
        warmup_epochs: int = 0
    ) -> float:
        """
        Scale learning rate for multi-GPU training
        
        Args:
            base_lr: Base learning rate
            scaling_rule: 'linear', 'sqrt', or 'none'
            warmup_epochs: Number of warmup epochs for large batch training
            
        Returns:
            Scaled learning rate
        """
        if scaling_rule == 'none' or self.num_gpus == 1:
            return base_lr
        elif scaling_rule == 'linear':
            scaled_lr = base_lr * self.num_gpus
        elif scaling_rule == 'sqrt':
            scaled_lr = base_lr * math.sqrt(self.num_gpus)
        else:
            raise ValueError(f"Unknown scaling rule: {scaling_rule}")
            
        logger.info(f"Learning rate scaled from {base_lr} to {scaled_lr} using {scaling_rule} rule")
        
        if warmup_epochs > 0 and scaled_lr > base_lr:
            logger.info(f"Recommended warmup epochs: {warmup_epochs}")
            
        return scaled_lr
    
    def barrier(self):
        """Synchronize all processes"""
        if self.distributed and dist.is_initialized():
            dist.barrier()
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)"""
        return self.rank == 0
    
    def get_gpu_memory_stats(self) -> dict:
        """Get GPU memory statistics for all used GPUs"""
        stats = {}
        for i, gpu_id in enumerate(self.gpu_ids):
            if torch.cuda.is_available():
                stats[f'gpu_{gpu_id}'] = {
                    'allocated': torch.cuda.memory_allocated(gpu_id) / 1024**3,  # GB
                    'cached': torch.cuda.memory_reserved(gpu_id) / 1024**3,     # GB
                    'max_allocated': torch.cuda.max_memory_allocated(gpu_id) / 1024**3,  # GB
                }
        return stats


def setup_multi_gpu_training(
    model: nn.Module,
    gpu_ids: Optional[Union[List[int], str]] = None,
    distributed: bool = False,
    backend: str = 'nccl'
) -> Tuple[nn.Module, MultiGPUWrapper]:
    """
    Convenience function to setup multi-GPU training
    
    Args:
        model: Model to wrap
        gpu_ids: GPU IDs to use
        distributed: Whether to use distributed training
        backend: Distributed backend
        
    Returns:
        Tuple of (wrapped_model, gpu_wrapper)
    """
    wrapper = MultiGPUWrapper(gpu_ids=gpu_ids, distributed=distributed, backend=backend)
    
    if distributed:
        wrapper.setup_distributed()
        
    wrapped_model = wrapper.wrap_model(model)
    
    return wrapped_model, wrapper


def optimize_dataloader_for_multigpu(
    dataloader: DataLoader,
    wrapper: MultiGPUWrapper,
    num_workers_multiplier: int = 2
) -> DataLoader:
    """
    Optimize DataLoader settings for multi-GPU training
    
    Args:
        dataloader: Original DataLoader
        wrapper: MultiGPUWrapper instance
        num_workers_multiplier: Multiplier for number of workers per GPU
        
    Returns:
        Optimized DataLoader
    """
    # Calculate optimal number of workers
    optimal_workers = min(
        wrapper.num_gpus * num_workers_multiplier,
        os.cpu_count() or 4
    )
    
    # Create distributed sampler if needed
    distributed_sampler = wrapper.create_distributed_sampler(dataloader.dataset)
    
    # Create new DataLoader with optimized settings
    # Preserve the original collate_fn to avoid default_collate issues with variable-sized fields
    original_collate = getattr(dataloader, 'collate_fn', None)

    optimized_loader = DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=(distributed_sampler is None),  # Don't shuffle if using distributed sampler
        sampler=distributed_sampler,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True if optimal_workers > 0 else False,
        drop_last=True,
        collate_fn=original_collate
    )
    
    logger.info(f"DataLoader optimized for multi-GPU:")
    logger.info(f"  - Workers: {optimal_workers}")
    logger.info(f"  - Distributed sampler: {distributed_sampler is not None}")
    logger.info(f"  - Pin memory: True")
    logger.info(f"  - Persistent workers: {optimal_workers > 0}")
    
    return optimized_loader


def log_gpu_memory_usage(wrapper: MultiGPUWrapper, prefix: str = ""):
    """Log current GPU memory usage"""
    if wrapper.is_main_process():
        stats = wrapper.get_gpu_memory_stats()
        for gpu_name, gpu_stats in stats.items():
            logger.info(
                f"{prefix}{gpu_name}: "
                f"Allocated: {gpu_stats['allocated']:.2f}GB, "
                f"Cached: {gpu_stats['cached']:.2f}GB, "
                f"Max: {gpu_stats['max_allocated']:.2f}GB"
            )


class GradientAccumulator:
    """Helper class for gradient accumulation in multi-GPU training"""
    
    def __init__(self, accumulation_steps: int, wrapper: MultiGPUWrapper):
        self.accumulation_steps = accumulation_steps
        self.wrapper = wrapper
        self.step_count = 0
        
    def should_step(self) -> bool:
        """Check if optimizer should step"""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by accumulation steps"""
        return loss / self.accumulation_steps
    
    def reset(self):
        """Reset step counter"""
        self.step_count = 0