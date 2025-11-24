"""Statistics calculation module for Training Surgeon."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
from .utils import safe_tensor_stats


@dataclass
class LayerStatistics:
    """Statistics for a single layer."""
    layer_name: str
    step: int
    
    # Activation statistics
    activation_mean: Optional[float] = None
    activation_std: Optional[float] = None
    activation_min: Optional[float] = None
    activation_max: Optional[float] = None
    activation_numel: Optional[int] = None
    
    # Gradient statistics
    gradient_mean: Optional[float] = None
    gradient_std: Optional[float] = None
    gradient_min: Optional[float] = None
    gradient_max: Optional[float] = None
    gradient_numel: Optional[int] = None
    
    # Parameter statistics
    parameter_mean: Optional[float] = None
    parameter_std: Optional[float] = None
    parameter_min: Optional[float] = None
    parameter_max: Optional[float] = None
    parameter_numel: Optional[int] = None
    
    def has_activation_stats(self) -> bool:
        """Check if activation statistics are available."""
        return self.activation_mean is not None
    
    def has_gradient_stats(self) -> bool:
        """Check if gradient statistics are available."""
        return self.gradient_mean is not None
    
    def has_parameter_stats(self) -> bool:
        """Check if parameter statistics are available."""
        return self.parameter_mean is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'layer_name': self.layer_name,
            'step': self.step,
            'activation': {
                'mean': self.activation_mean,
                'std': self.activation_std,
                'min': self.activation_min,
                'max': self.activation_max,
                'numel': self.activation_numel
            } if self.has_activation_stats() else None,
            'gradient': {
                'mean': self.gradient_mean,
                'std': self.gradient_std,
                'min': self.gradient_min,
                'max': self.gradient_max,
                'numel': self.gradient_numel
            } if self.has_gradient_stats() else None,
            'parameter': {
                'mean': self.parameter_mean,
                'std': self.parameter_std,
                'min': self.parameter_min,
                'max': self.parameter_max,
                'numel': self.parameter_numel
            } if self.has_parameter_stats() else None
        }


class StatisticsCalculator:
    """Calculator for layer statistics."""
    
    def __init__(self):
        self.statistics: Dict[int, Dict[str, LayerStatistics]] = {}
        self.current_step = 0
    
    def set_current_step(self, step: int):
        """Set the current training step."""
        self.current_step = step
        if step not in self.statistics:
            self.statistics[step] = {}
    
    def record_activation_stats(self, layer_name: str, tensor: torch.Tensor):
        """Record activation statistics for a layer."""
        stats = safe_tensor_stats(tensor)
        
        if self.current_step not in self.statistics:
            self.statistics[self.current_step] = {}
        
        if layer_name not in self.statistics[self.current_step]:
            self.statistics[self.current_step][layer_name] = LayerStatistics(
                layer_name=layer_name,
                step=self.current_step
            )
        
        layer_stats = self.statistics[self.current_step][layer_name]
        layer_stats.activation_mean = stats['mean']
        layer_stats.activation_std = stats['std']
        layer_stats.activation_min = stats['min']
        layer_stats.activation_max = stats['max']
        layer_stats.activation_numel = stats['numel']
    
    def record_gradient_stats(self, layer_name: str, tensor: torch.Tensor):
        """Record gradient statistics for a layer."""
        if tensor is None:
            return
        
        stats = safe_tensor_stats(tensor)
        
        if self.current_step not in self.statistics:
            self.statistics[self.current_step] = {}
        
        if layer_name not in self.statistics[self.current_step]:
            self.statistics[self.current_step][layer_name] = LayerStatistics(
                layer_name=layer_name,
                step=self.current_step
            )
        
        layer_stats = self.statistics[self.current_step][layer_name]
        layer_stats.gradient_mean = stats['mean']
        layer_stats.gradient_std = stats['std']
        layer_stats.gradient_min = stats['min']
        layer_stats.gradient_max = stats['max']
        layer_stats.gradient_numel = stats['numel']
    
    def record_parameter_stats(self, layer_name: str, tensor: torch.Tensor):
        """Record parameter statistics for a layer."""
        stats = safe_tensor_stats(tensor)
        
        if self.current_step not in self.statistics:
            self.statistics[self.current_step] = {}
        
        if layer_name not in self.statistics[self.current_step]:
            self.statistics[self.current_step][layer_name] = LayerStatistics(
                layer_name=layer_name,
                step=self.current_step
            )
        
        layer_stats = self.statistics[self.current_step][layer_name]
        layer_stats.parameter_mean = stats['mean']
        layer_stats.parameter_std = stats['std']
        layer_stats.parameter_min = stats['min']
        layer_stats.parameter_max = stats['max']
        layer_stats.parameter_numel = stats['numel']
    
    def get_step_statistics(self, step: int) -> Dict[str, LayerStatistics]:
        """Get all statistics for a specific step."""
        return self.statistics.get(step, {})
    
    def get_all_statistics(self) -> Dict[int, Dict[str, LayerStatistics]]:
        """Get all recorded statistics."""
        return self.statistics
    
    def get_layer_names(self, step: Optional[int] = None) -> List[str]:
        """Get all layer names, optionally for a specific step."""
        if step is not None:
            return list(self.statistics.get(step, {}).keys())
        
        # Get all unique layer names across all steps
        layer_names = set()
        for step_stats in self.statistics.values():
            layer_names.update(step_stats.keys())
        
        return sorted(list(layer_names))
    
    def clear_step(self, step: int):
        """Clear statistics for a specific step."""
        if step in self.statistics:
            del self.statistics[step]
    
    def clear_all(self):
        """Clear all statistics."""
        self.statistics.clear()
        self.current_step = 0