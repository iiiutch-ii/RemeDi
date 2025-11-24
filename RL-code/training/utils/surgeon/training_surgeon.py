"""Main Training Surgeon class for monitoring PyTorch model training."""

import os
import time
import json
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import torch
import torch.nn as nn

from .statistics import StatisticsCalculator, LayerStatistics
from .hooks import HookManager
from .utils import is_rank0, ensure_dir, format_number


class TrainingSurgeon:
    """
    Main class for monitoring model training statistics.
    
    Monitors activations, gradients, and parameters during training.
    Only operates on rank 0 in distributed training to avoid overhead.
    """
    
    def __init__(
        self,
        model: nn.Module,
        enabled: bool = True,
        record_every: int = 1,
        output_dir: str = "./training_surgeon_logs",
        monitor_activations: bool = True,
        monitor_gradients: bool = True,
        monitor_parameters: bool = True,
        activation_hook_types: Optional[List[type]] = None,
        save_memory: bool = True
    ):
        """
        Initialize Training Surgeon.
        
        Args:
            model: PyTorch model to monitor
            enabled: Whether to enable monitoring initially
            record_every: Record statistics every N steps
            output_dir: Directory to save outputs
            monitor_activations: Whether to monitor layer activations
            monitor_gradients: Whether to monitor gradients
            monitor_parameters: Whether to monitor parameters
            activation_hook_types: Specific module types to hook for activations
            save_memory: Whether to use memory-efficient operations
        """
        self.model = model
        self.enabled = enabled and is_rank0()  # Only enable on rank 0
        self.record_every = record_every
        self.output_dir = Path(output_dir)
        self.monitor_activations = monitor_activations
        self.monitor_gradients = monitor_gradients
        self.monitor_parameters = monitor_parameters
        self.save_memory = save_memory
        
        # Core components
        self.statistics_calculator = StatisticsCalculator()
        self.hook_manager = HookManager(self.statistics_calculator)
        
        # State tracking
        self.current_step = 0
        self.last_recorded_step = -1
        self.is_context_manager = False
        
        # Initialize if enabled
        if self.enabled:
            self._initialize()
    
    def _initialize(self):
        """Initialize the surgeon (only on rank 0)."""
        if not self.enabled:
            return
        
        # Create output directory
        ensure_dir(str(self.output_dir))
        
        # Register hooks
        self.hook_manager.register_all_hooks(
            self.model,
            monitor_activations=self.monitor_activations,
            monitor_gradients=self.monitor_gradients,
            monitor_parameters=self.monitor_parameters,
            activation_hook_types=None  # Use defaults
        )
        
        # Enable hooks
        self.hook_manager.enable()
        
        print(f"Training Surgeon initialized (rank 0 only)")
        hook_summary = self.hook_manager.get_hook_summary()
        print(f"Registered hooks: {hook_summary}")
    
    def enable(self):
        """Enable monitoring."""
        if not is_rank0():
            return
        
        if not self.enabled:
            self.enabled = True
            self._initialize()
        else:
            self.hook_manager.enable()
    
    def disable(self):
        """Disable monitoring."""
        self.enabled = False
        self.hook_manager.disable()
    
    def step(self, step: Optional[int] = None):
        """
        Notify surgeon of a training step.
        
        Args:
            step: Optional step number. If None, auto-increment.
        """
        if not self.enabled:
            return
        
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Update statistics calculator
        self.statistics_calculator.set_current_step(self.current_step)
        
        # Record statistics if it's time
        if self._should_record():
            self._record_current_step()
    
    def _should_record(self) -> bool:
        """Check if we should record statistics for current step."""
        return (
            self.enabled and 
            self.current_step % self.record_every == 0 and
            self.current_step != self.last_recorded_step
        )
    
    def _record_current_step(self):
        """Record statistics for the current step."""
        if not self.enabled:
            return
        
        # Capture parameter statistics manually
        if self.monitor_parameters:
            self.hook_manager.capture_parameter_stats()
        
        self.last_recorded_step = self.current_step
    
    def save_report(
        self, 
        step: Optional[int] = None, 
        include_plots: bool = True,
        include_text: bool = True
    ):
        """
        Save statistics report for a specific step.
        
        Args:
            step: Step to save report for. If None, use current step.
            include_plots: Whether to generate and save plots
            include_text: Whether to generate and save text report
        """
        if not self.enabled:
            return
        
        if step is None:
            step = self.current_step
        
        step_stats = self.statistics_calculator.get_step_statistics(step)
        if not step_stats:
            print(f"No statistics available for step {step}")
            return
        
        # Save text report
        if include_text:
            self._save_text_report(step, step_stats)
        
        # Save plots
        if include_plots:
            self._save_plots(step, step_stats)
        
        print(f"Saved training surgeon report for step {step} to {self.output_dir}")
    
    def _save_text_report(self, step: int, step_stats: Dict[str, LayerStatistics]):
        """Save detailed text report."""
        report_path = self.output_dir / f"training_stats_step{step:06d}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Training Surgeon Report - Step {step}\n")
            f.write("=" * 50 + "\n\n")
            
            # Sort layers by name for consistent ordering
            sorted_layers = sorted(step_stats.items())
            
            for layer_name, stats in sorted_layers:
                f.write(f"Layer: {layer_name}\n")
                f.write("-" * 30 + "\n")
                
                # Activation statistics
                if stats.has_activation_stats():
                    f.write("Activations:\n")
                    f.write(f"  Mean: {format_number(stats.activation_mean)}\n")
                    f.write(f"  Std:  {format_number(stats.activation_std)}\n")
                    f.write(f"  Min:  {format_number(stats.activation_min)}\n")
                    f.write(f"  Max:  {format_number(stats.activation_max)}\n")
                    f.write(f"  Size: {stats.activation_numel}\n\n")
                
                # Gradient statistics
                if stats.has_gradient_stats():
                    f.write("Gradients:\n")
                    f.write(f"  Mean: {format_number(stats.gradient_mean)}\n")
                    f.write(f"  Std:  {format_number(stats.gradient_std)}\n")
                    f.write(f"  Min:  {format_number(stats.gradient_min)}\n")
                    f.write(f"  Max:  {format_number(stats.gradient_max)}\n")
                    f.write(f"  Size: {stats.gradient_numel}\n\n")
                
                # Parameter statistics
                if stats.has_parameter_stats():
                    f.write("Parameters:\n")
                    f.write(f"  Mean: {format_number(stats.parameter_mean)}\n")
                    f.write(f"  Std:  {format_number(stats.parameter_std)}\n")
                    f.write(f"  Min:  {format_number(stats.parameter_min)}\n")
                    f.write(f"  Max:  {format_number(stats.parameter_max)}\n")
                    f.write(f"  Size: {stats.parameter_numel}\n\n")
                
                f.write("\n")
            
            # Summary
            f.write("Summary\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total layers monitored: {len(step_stats)}\n")
            
            activation_layers = sum(1 for s in step_stats.values() if s.has_activation_stats())
            gradient_layers = sum(1 for s in step_stats.values() if s.has_gradient_stats())
            parameter_layers = sum(1 for s in step_stats.values() if s.has_parameter_stats())
            
            f.write(f"Layers with activation stats: {activation_layers}\n")
            f.write(f"Layers with gradient stats: {gradient_layers}\n")
            f.write(f"Layers with parameter stats: {parameter_layers}\n")
    
    def _save_plots(self, step: int, step_stats: Dict[str, LayerStatistics]):
        """Save visualization plots."""
        try:
            from .visualization import StatisticsVisualizer
            visualizer = StatisticsVisualizer(self.output_dir)
            visualizer.plot_step_statistics(step, step_stats)
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def save_json_report(self, step: Optional[int] = None):
        """Save statistics as JSON for programmatic access."""
        if not self.enabled:
            return
        
        if step is None:
            step = self.current_step
        
        step_stats = self.statistics_calculator.get_step_statistics(step)
        if not step_stats:
            return
        
        # Convert to serializable format
        json_data = {
            'step': step,
            'timestamp': time.time(),
            'layers': {name: stats.to_dict() for name, stats in step_stats.items()}
        }
        
        json_path = self.output_dir / f"training_stats_step{step:06d}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def cleanup(self):
        """Clean up resources."""
        self.hook_manager.remove_all_hooks()
        if self.save_memory:
            self.statistics_calculator.clear_all()
    
    # Context manager support
    def __enter__(self):
        """Enter context manager."""
        self.is_context_manager = True
        if self.enabled:
            self.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.enabled:
            # Save final report
            if self.current_step > self.last_recorded_step:
                self._record_current_step()
                self.save_report()
        
        self.cleanup()
        self.is_context_manager = False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of surgeon state."""
        return {
            'enabled': self.enabled,
            'is_rank0': is_rank0(),
            'current_step': self.current_step,
            'last_recorded_step': self.last_recorded_step,
            'record_every': self.record_every,
            'output_dir': str(self.output_dir),
            'monitoring': {
                'activations': self.monitor_activations,
                'gradients': self.monitor_gradients,
                'parameters': self.monitor_parameters
            },
            'hooks': self.hook_manager.get_hook_summary() if self.enabled else {}
        }