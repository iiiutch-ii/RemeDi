"""Hook system for monitoring activations, gradients, and parameters."""

from typing import Dict, List, Callable, Optional, Any
import torch
import torch.nn as nn
from .statistics import StatisticsCalculator
from .utils import get_layer_name, memory_efficient_hook_removal


class HookManager:
    """Manages all hooks for the Training Surgeon."""
    
    def __init__(self, statistics_calculator: StatisticsCalculator):
        self.statistics_calculator = statistics_calculator
        self.hooks: List[Any] = []
        self.activation_hooks: Dict[str, Any] = {}
        self.gradient_hooks: Dict[str, Any] = {}
        self.parameter_hooks: Dict[str, Any] = {}
        self.enabled = False
    
    def enable(self):
        """Enable hook recording."""
        self.enabled = True
    
    def disable(self):
        """Disable hook recording."""
        self.enabled = False
    
    def register_activation_hooks(self, model: nn.Module, hook_types: Optional[List[type]] = None):
        """Register forward hooks to capture activations."""
        if hook_types is None:
            # Default hook types for common layers
            hook_types = [
                nn.Linear,
                nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.LSTM, nn.GRU,
                nn.TransformerEncoderLayer, nn.TransformerDecoderLayer,
                nn.MultiheadAttention,
                nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid
            ]
        
        def create_activation_hook(name: str):
            def hook_fn(module, input, output):
                if not self.enabled:
                    return
                
                # Handle tuple outputs (e.g., LSTM)
                if isinstance(output, tuple):
                    output = output[0]
                
                if isinstance(output, torch.Tensor):
                    self.statistics_calculator.record_activation_stats(name, output)
            
            return hook_fn
        
        for name, module in model.named_modules():
            if any(isinstance(module, hook_type) for hook_type in hook_types):
                clean_name = get_layer_name(module, name)
                hook = module.register_forward_hook(create_activation_hook(clean_name))
                self.hooks.append(hook)
                self.activation_hooks[clean_name] = hook
    
    def register_gradient_hooks(self, model: nn.Module):
        """Register backward hooks to capture gradients."""
        
        def create_gradient_hook(name: str, param: nn.Parameter):
            def hook_fn(grad):
                if not self.enabled or grad is None:
                    return grad
                
                self.statistics_calculator.record_gradient_stats(name, grad)
                return grad
            
            return hook_fn
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                clean_name = get_layer_name(None, name)
                hook = param.register_hook(create_gradient_hook(clean_name, param))
                self.hooks.append(hook)
                self.gradient_hooks[clean_name] = hook
    
    def register_parameter_hooks(self, model: nn.Module):
        """Register hooks to monitor parameter values."""
        # Parameter values are captured during forward pass
        for name, param in model.named_parameters():
            clean_name = get_layer_name(None, name)
            # We'll capture parameter stats in the main surgeon class
            self.parameter_hooks[clean_name] = param
    
    def capture_parameter_stats(self):
        """Manually capture parameter statistics."""
        if not self.enabled:
            return
        
        for name, param in self.parameter_hooks.items():
            if isinstance(param, nn.Parameter):
                self.statistics_calculator.record_parameter_stats(name, param)
    
    def register_all_hooks(self, model: nn.Module, 
                          monitor_activations: bool = True,
                          monitor_gradients: bool = True,
                          monitor_parameters: bool = True,
                          activation_hook_types: Optional[List[type]] = None):
        """Register all types of hooks."""
        
        if monitor_activations:
            self.register_activation_hooks(model, activation_hook_types)
        
        if monitor_gradients:
            self.register_gradient_hooks(model)
        
        if monitor_parameters:
            self.register_parameter_hooks(model)
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        memory_efficient_hook_removal(self.hooks)
        self.activation_hooks.clear()
        self.gradient_hooks.clear()
        self.parameter_hooks.clear()
    
    def get_hook_summary(self) -> Dict[str, int]:
        """Get summary of registered hooks."""
        return {
            'total_hooks': len(self.hooks),
            'activation_hooks': len(self.activation_hooks),
            'gradient_hooks': len(self.gradient_hooks),
            'parameter_hooks': len(self.parameter_hooks)
        }


class ActivationHook:
    """Standalone activation hook class."""
    
    def __init__(self, module: nn.Module, name: str, 
                 callback: Callable[[str, torch.Tensor], None]):
        self.module = module
        self.name = name
        self.callback = callback
        self.enabled = True
        self._hook = None
        self.register()
    
    def register(self):
        """Register the hook."""
        def hook_fn(module, input, output):
            if not self.enabled:
                return
            
            if isinstance(output, tuple):
                output = output[0]
            
            if isinstance(output, torch.Tensor):
                self.callback(self.name, output)
        
        self._hook = self.module.register_forward_hook(hook_fn)
    
    def enable(self):
        """Enable the hook."""
        self.enabled = True
    
    def disable(self):
        """Disable the hook."""
        self.enabled = False
    
    def remove(self):
        """Remove the hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


class GradientHook:
    """Standalone gradient hook class."""
    
    def __init__(self, parameter: nn.Parameter, name: str,
                 callback: Callable[[str, torch.Tensor], None]):
        self.parameter = parameter
        self.name = name
        self.callback = callback
        self.enabled = True
        self._hook = None
        self.register()
    
    def register(self):
        """Register the hook."""
        def hook_fn(grad):
            if not self.enabled or grad is None:
                return grad
            
            self.callback(self.name, grad)
            return grad
        
        self._hook = self.parameter.register_hook(hook_fn)
    
    def enable(self):
        """Enable the hook."""
        self.enabled = True
    
    def disable(self):
        """Disable the hook."""
        self.enabled = False
    
    def remove(self):
        """Remove the hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None