"""Visualization module for Training Surgeon statistics."""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

from .statistics import LayerStatistics
from .utils import format_number

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class StatisticsVisualizer:
    """Visualizes training statistics using matplotlib."""
    
    def __init__(self, output_dir: Path, style: str = 'default'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.style = style
        
        # Set up matplotlib
        plt.style.use(style)
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 10
        
        # Colors for different metrics
        self.colors = {
            'activation': {'mean': '#1f77b4', 'std': '#aec7e8', 'min': '#ff7f0e', 'max': '#ffbb78'},
            'gradient': {'mean': '#2ca02c', 'std': '#98df8a', 'min': '#d62728', 'max': '#ff9896'},
            'parameter': {'mean': '#9467bd', 'std': '#c5b0d5', 'min': '#8c564b', 'max': '#c49c94'}
        }
    
    def plot_step_statistics(
        self, 
        step: int, 
        step_stats: Dict[str, LayerStatistics],
        save_format: str = 'png',
        dpi: int = 150
    ):
        """
        Plot statistics for a single training step.
        
        Args:
            step: Training step number
            step_stats: Layer statistics for the step
            save_format: Image format ('png', 'pdf', 'svg')
            dpi: Image DPI
        """
        if not step_stats:
            print(f"No statistics to plot for step {step}")
            return
        
        # Sort layers by name for consistent ordering
        sorted_layers = sorted(step_stats.items())
        layer_names = [name for name, _ in sorted_layers]
        layer_indices = list(range(len(layer_names)))
        
        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Training Statistics - Step {step}', fontsize=16, fontweight='bold')
        
        # Extract data for each metric type
        activation_data = self._extract_metric_data(sorted_layers, 'activation')
        gradient_data = self._extract_metric_data(sorted_layers, 'gradient')
        parameter_data = self._extract_metric_data(sorted_layers, 'parameter')
        
        # Plot activations
        self._plot_metric_row(axes[0], layer_indices, layer_names, activation_data, 
                             'Activations', self.colors['activation'])
        
        # Plot gradients
        self._plot_metric_row(axes[1], layer_indices, layer_names, gradient_data,
                             'Gradients', self.colors['gradient'])
        
        # Plot parameters
        self._plot_metric_row(axes[2], layer_indices, layer_names, parameter_data,
                             'Parameters', self.colors['parameter'])
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save plot
        plot_path = self.output_dir / f"training_plots_step{step:06d}.{save_format}"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved plots to {plot_path}")
    
    def _extract_metric_data(
        self, 
        sorted_layers: List[Tuple[str, LayerStatistics]], 
        metric_type: str
    ) -> Dict[str, List[float]]:
        """Extract data for a specific metric type (activation, gradient, parameter)."""
        data = {'mean': [], 'std': [], 'min': [], 'max': []}
        
        for _, stats in sorted_layers:
            if metric_type == 'activation' and stats.has_activation_stats():
                data['mean'].append(stats.activation_mean)
                data['std'].append(stats.activation_std)
                data['min'].append(stats.activation_min)
                data['max'].append(stats.activation_max)
            elif metric_type == 'gradient' and stats.has_gradient_stats():
                data['mean'].append(stats.gradient_mean)
                data['std'].append(stats.gradient_std)
                data['min'].append(stats.gradient_min)
                data['max'].append(stats.gradient_max)
            elif metric_type == 'parameter' and stats.has_parameter_stats():
                data['mean'].append(stats.parameter_mean)
                data['std'].append(stats.parameter_std)
                data['min'].append(stats.parameter_min)
                data['max'].append(stats.parameter_max)
            else:
                # Fill with None for missing data
                data['mean'].append(None)
                data['std'].append(None)
                data['min'].append(None)
                data['max'].append(None)
        
        return data
    
    def _plot_metric_row(
        self, 
        axes_row, 
        layer_indices: List[int], 
        layer_names: List[str],
        data: Dict[str, List[float]], 
        metric_name: str, 
        colors: Dict[str, str]
    ):
        """Plot a row of metric subplots (mean, std, min, max)."""
        
        metrics = ['mean', 'std', 'min', 'max']
        titles = ['Mean', 'Standard Deviation', 'Minimum', 'Maximum']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes_row[i]
            values = data[metric]
            
            # Filter out None values
            valid_indices = [j for j, v in enumerate(values) if v is not None]
            valid_values = [values[j] for j in valid_indices]
            valid_names = [layer_names[j] for j in valid_indices]
            
            if not valid_values:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric_name} {title}')
                continue
            
            # Create bar plot
            bars = ax.bar(range(len(valid_values)), valid_values, 
                         color=colors[metric], alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_title(f'{metric_name} {title}')
            ax.set_ylabel('Value')
            
            # Set x-axis labels
            if len(valid_names) <= 20:
                ax.set_xticks(range(len(valid_names)))
                ax.set_xticklabels(valid_names, rotation=45, ha='right')
            else:
                # For many layers, show fewer labels
                step_size = max(1, len(valid_names) // 10)
                tick_positions = list(range(0, len(valid_names), step_size))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([valid_names[j] for j in tick_positions], 
                                  rotation=45, ha='right')
            
            # Add value labels on bars if not too many
            if len(valid_values) <= 15:
                for bar, value in zip(bars, valid_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           format_number(value), ha='center', va='bottom', fontsize=8)
            
            # Grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
            
            # Handle scientific notation for small values
            if any(abs(v) < 1e-3 for v in valid_values if v != 0):
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))
    
    def plot_multi_step_comparison(
        self,
        steps: List[int],
        all_step_stats: Dict[int, Dict[str, LayerStatistics]],
        layers_to_plot: Optional[List[str]] = None,
        save_format: str = 'png',
        dpi: int = 150
    ):
        """
        Plot comparison across multiple steps.
        
        Args:
            steps: List of step numbers to compare
            all_step_stats: Dictionary of step -> layer stats
            layers_to_plot: Specific layers to plot (if None, plot all)
            save_format: Image format
            dpi: Image DPI
        """
        if not steps or not all_step_stats:
            print("No data available for multi-step comparison")
            return
        
        # Get common layers across all steps
        if layers_to_plot is None:
            layer_sets = [set(all_step_stats[step].keys()) for step in steps if step in all_step_stats]
            if not layer_sets:
                return
            common_layers = sorted(list(set.intersection(*layer_sets)))
        else:
            common_layers = layers_to_plot
        
        if not common_layers:
            print("No common layers found across steps")
            return
        
        # Limit number of layers for readability
        if len(common_layers) > 12:
            common_layers = common_layers[:12]
            print(f"Limiting to first 12 layers for readability")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Multi-Step Comparison (Steps: {", ".join(map(str, steps))})', 
                    fontsize=16, fontweight='bold')
        
        # Plot trends for different metrics
        self._plot_multi_step_metric(axes[0, 0], steps, all_step_stats, common_layers, 
                                   'activation', 'mean', 'Activation Mean')
        self._plot_multi_step_metric(axes[0, 1], steps, all_step_stats, common_layers, 
                                   'gradient', 'mean', 'Gradient Mean')
        self._plot_multi_step_metric(axes[1, 0], steps, all_step_stats, common_layers, 
                                   'parameter', 'mean', 'Parameter Mean')
        self._plot_multi_step_metric(axes[1, 1], steps, all_step_stats, common_layers, 
                                   'activation', 'std', 'Activation Std')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save plot
        steps_str = "_".join(map(str, steps))
        plot_path = self.output_dir / f"training_comparison_steps_{steps_str}.{save_format}"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved multi-step comparison to {plot_path}")
    
    def _plot_multi_step_metric(
        self,
        ax,
        steps: List[int],
        all_step_stats: Dict[int, Dict[str, LayerStatistics]],
        layers: List[str],
        metric_type: str,
        stat_type: str,
        title: str
    ):
        """Plot a specific metric across multiple steps."""
        
        # Colors for different layers
        colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
        
        for i, layer in enumerate(layers):
            values = []
            valid_steps = []
            
            for step in steps:
                if step in all_step_stats and layer in all_step_stats[step]:
                    stats = all_step_stats[step][layer]
                    
                    if metric_type == 'activation' and stats.has_activation_stats():
                        value = getattr(stats, f'activation_{stat_type}')
                    elif metric_type == 'gradient' and stats.has_gradient_stats():
                        value = getattr(stats, f'gradient_{stat_type}')
                    elif metric_type == 'parameter' and stats.has_parameter_stats():
                        value = getattr(stats, f'parameter_{stat_type}')
                    else:
                        continue
                    
                    values.append(value)
                    valid_steps.append(step)
            
            if values:
                ax.plot(valid_steps, values, marker='o', label=layer, 
                       color=colors[i], linewidth=2, markersize=4)
        
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Legend with limited entries
        if len(layers) <= 8:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.text(0.02, 0.98, f'{len(layers)} layers', transform=ax.transAxes, 
                   va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Handle scientific notation
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))