"""
Shared Training Utilities

Common functions used by both BaseTrainer and BaseRLTrainer to avoid
code duplication. These are standalone functions (not methods) so both
trainer hierarchies can use them without an inheritance relationship.
"""

import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from vlex_ria.core.utils import get_logger

logger = get_logger(__name__)


def create_adamw_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    adam_epsilon: float = 1e-8,
) -> AdamW:
    """
    Create AdamW optimizer with proper weight decay separation.
    
    Bias, normalization, and embedding parameters are excluded from
    weight decay following best practices from GPT/LLaMA training.
    
    Args:
        model: Model whose parameters to optimize
        learning_rate: Base learning rate
        weight_decay: Weight decay for non-excluded parameters
        adam_beta1: Adam beta1
        adam_beta2: Adam beta2
        adam_epsilon: Adam epsilon
        
    Returns:
        Configured AdamW optimizer
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return AdamW(
        [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )


def save_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    **extra_fields,
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        path: Full path for the checkpoint file
        model: Model to save
        optimizer: Optimizer to save
        scheduler: LR scheduler to save
        global_step: Current training step
        **extra_fields: Additional fields (e.g. best_val_loss, best_reward, config)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step,
        **extra_fields,
    }
    
    torch.save(checkpoint, path)
    logger.info(f"  Checkpoint saved: {path}")


def load_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Restores model, optimizer, and scheduler state. Returns the full
    checkpoint dict so callers can read extra fields (best_val_loss, etc.).
    
    Args:
        path: Checkpoint file path
        model: Model to restore into
        optimizer: Optimizer to restore into
        scheduler: Scheduler to restore into
        device: Device to map tensors to
        
    Returns:
        Full checkpoint dict (callers read extra fields like global_step)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded checkpoint from {path}, step {checkpoint.get('global_step', '?')}")
    
    return checkpoint
