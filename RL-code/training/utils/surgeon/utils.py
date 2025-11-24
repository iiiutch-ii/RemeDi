"""Utility functions for Training Surgeon."""

import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
from pathlib import Path


def is_rank0() -> bool:
    """Check if current process is rank 0 in distributed training."""
    if not dist.is_available():
        return True
    
    if not dist.is_initialized():
        return True
    
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of current process."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

@torch.no_grad()
def safe_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Safely compute tensor statistics, handling edge cases."""
    if tensor.numel() == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'numel': 0
        }
    
    # Move to CPU if needed to avoid memory issues on GPU
    if tensor.is_cuda and tensor.numel() > 1e6:  # Large tensors
        tensor_cpu = tensor.detach().cpu()
    else:
        tensor_cpu = tensor.detach()
    
    # Handle different dtypes
    if tensor_cpu.dtype in [torch.float16, torch.bfloat16]:
        tensor_cpu = tensor_cpu.float()
    
    try:
        stats = {
            'mean': float(tensor_cpu.mean()),
            'std': float(tensor_cpu.std()),
            'min': float(tensor_cpu.min()),
            'max': float(tensor_cpu.max()),
            'numel': int(tensor.numel())
        }
    except RuntimeError:
        # Fallback for problematic tensors
        flat = tensor_cpu.flatten()
        stats = {
            'mean': float(flat.mean()) if flat.numel() > 0 else 0.0,
            'std': float(flat.std()) if flat.numel() > 1 else 0.0,
            'min': float(flat.min()) if flat.numel() > 0 else 0.0,
            'max': float(flat.max()) if flat.numel() > 0 else 0.0,
            'numel': int(tensor.numel())
        }
    
    return stats


def ensure_dir(path: str) -> str:
    """Ensure directory exists and return the path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def format_number(num: float, precision: int = 6) -> str:
    """Format a number with appropriate precision."""
    if abs(num) < 1e-6:
        return f"{num:.2e}"
    elif abs(num) < 1e-3:
        return f"{num:.6f}"
    elif abs(num) < 1:
        return f"{num:.4f}"
    else:
        return f"{num:.3f}"


def get_layer_name(module: torch.nn.Module, name: str = "") -> str:
    """Get a clean layer name for display."""
    if name == "":
        return type(module).__name__
    
    # Clean up the name
    clean_name = name.replace("_fsdp_wrapped_module", "")
    clean_name = clean_name.replace("._orig_mod", "")
    
    return clean_name


def memory_efficient_hook_removal(hooks: list) -> None:
    """Safely remove hooks with memory cleanup."""
    for hook in hooks:
        if hasattr(hook, 'remove'):
            try:
                hook.remove()
            except (RuntimeError, AttributeError):
                pass  # Hook may already be removed
    hooks.clear()