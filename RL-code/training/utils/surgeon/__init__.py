"""
Training Surgeon - Monitor PyTorch Model Training Statistics

A comprehensive tool for monitoring activations, gradients, and parameters
during model training. Optimized for distributed training environments.
"""

from .training_surgeon import TrainingSurgeon
from .statistics import LayerStatistics, StatisticsCalculator
from .hooks import HookManager, ActivationHook, GradientHook
from .visualization import StatisticsVisualizer
from .utils import is_rank0, get_rank, get_world_size

__version__ = "1.0.0"
__author__ = "Training Surgeon"

__all__ = [
    'TrainingSurgeon',
    'LayerStatistics', 
    'StatisticsCalculator',
    'HookManager',
    'ActivationHook',
    'GradientHook', 
    'StatisticsVisualizer',
    'is_rank0',
    'get_rank',
    'get_world_size'
]