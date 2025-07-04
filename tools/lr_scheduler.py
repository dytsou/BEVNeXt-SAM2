"""
Learning Rate Scheduler for BEVNeXt-SAM2 Fusion Training

Implements cosine annealing with warmup for stable training.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional


class CosineLRScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup.
    
    Features:
    - Linear warmup phase
    - Cosine annealing decay
    - Configurable minimum learning rate
    - Support for different parameter groups
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min: float = 0.0,
        warmup_t: int = 0,
        warmup_lr_init: float = 0.0,
        t_in_epochs: bool = True,
        noise_range_t: Optional[tuple] = None,
        noise_pct: float = 0.67,
        noise_std: float = 1.0,
        noise_seed: int = 42,
        last_epoch: int = -1
    ):
        """
        Initialize cosine LR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            t_initial: Total training epochs/steps
            lr_min: Minimum learning rate
            warmup_t: Warmup epochs/steps
            warmup_lr_init: Initial warmup learning rate
            t_in_epochs: If True, interpret t_initial as epochs
            noise_range_t: Add noise in specified epoch range
            noise_pct: Noise percentage
            noise_std: Noise standard deviation
            noise_seed: Random seed for noise
            last_epoch: Last epoch for resuming
        """
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
        
    def _get_lr(self, t: int) -> List[float]:
        """Get learning rate for given time step."""
        
        if t < self.warmup_t:
            # Warmup phase - linear increase
            lrs = []
            for base_lr in self.base_lrs:
                lr = self.warmup_lr_init + (base_lr - self.warmup_lr_init) * t / self.warmup_t
                lrs.append(lr)
        else:
            # Cosine annealing phase
            t_cosine = t - self.warmup_t
            t_max = self.t_initial - self.warmup_t
            
            lrs = []
            for base_lr in self.base_lrs:
                lr = self.lr_min + (base_lr - self.lr_min) * 0.5 * (
                    1 + math.cos(math.pi * t_cosine / t_max)
                )
                lrs.append(lr)
        
        # Add noise if specified
        if self.noise_range_t is not None:
            if self.noise_range_t[0] <= t <= self.noise_range_t[1]:
                lrs = self._add_noise(lrs, t)
        
        return lrs
        
    def _add_noise(self, lrs: List[float], t: int) -> List[float]:
        """Add noise to learning rates."""
        import random
        random.seed(self.noise_seed + t)
        
        noisy_lrs = []
        for lr in lrs:
            noise = random.gauss(0, self.noise_std) * self.noise_pct
            noisy_lr = lr * (1 + noise)
            noisy_lr = max(noisy_lr, self.lr_min)  # Ensure minimum LR
            noisy_lrs.append(noisy_lr)
            
        return noisy_lrs
        
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return self._get_lr(self.last_epoch)


class WarmupMultiStepLR(_LRScheduler):
    """
    Multi-step LR scheduler with warmup.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_epochs: int = 5,
        warmup_factor: float = 0.1,
        last_epoch: int = -1
    ):
        """
        Initialize multi-step LR scheduler with warmup.
        
        Args:
            optimizer: Optimizer to schedule
            milestones: Epochs to reduce learning rate
            gamma: Multiplicative factor of learning rate decay
            warmup_epochs: Number of warmup epochs
            warmup_factor: Warmup learning rate factor
            last_epoch: Last epoch for resuming
        """
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Multi-step phase
            return [
                base_lr * (self.gamma ** sum([self.last_epoch >= m for m in self.milestones]))
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate decay scheduler.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
        power: float = 0.9,
        last_epoch: int = -1
    ):
        """
        Initialize polynomial LR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            total_iters: Total training iterations
            power: Polynomial power
            last_epoch: Last epoch for resuming
        """
        self.total_iters = total_iters
        self.power = power
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        factor = (1 - self.last_epoch / self.total_iters) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class LinearLR(_LRScheduler):
    """
    Linear learning rate decay scheduler.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
        last_epoch: int = -1
    ):
        """
        Initialize linear LR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            total_iters: Total training iterations
            last_epoch: Last epoch for resuming
        """
        self.total_iters = total_iters
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        factor = max(0, 1 - self.last_epoch / self.total_iters)
        return [base_lr * factor for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict
) -> _LRScheduler:
    """
    Build learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_config: Scheduler configuration
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_config.get('type', 'cosine')
    
    if scheduler_type == 'cosine':
        return CosineLRScheduler(
            optimizer,
            t_initial=scheduler_config.get('t_initial', 100),
            lr_min=scheduler_config.get('lr_min', 0.0),
            warmup_t=scheduler_config.get('warmup_t', 5),
            warmup_lr_init=scheduler_config.get('warmup_lr_init', 0.0)
        )
    elif scheduler_type == 'multistep':
        return WarmupMultiStepLR(
            optimizer,
            milestones=scheduler_config.get('milestones', [30, 60, 90]),
            gamma=scheduler_config.get('gamma', 0.1),
            warmup_epochs=scheduler_config.get('warmup_epochs', 5)
        )
    elif scheduler_type == 'polynomial':
        return PolynomialLR(
            optimizer,
            total_iters=scheduler_config.get('total_iters', 100),
            power=scheduler_config.get('power', 0.9)
        )
    elif scheduler_type == 'linear':
        return LinearLR(
            optimizer,
            total_iters=scheduler_config.get('total_iters', 100)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")