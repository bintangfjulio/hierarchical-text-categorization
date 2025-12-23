"""
Visualization utilities for training results.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Dict, List
import pandas as pd


class Visualizer:
    """Visualize training results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save: bool = True
    ) -> None:
        """
        Plot training history.
        
        Args:
            history: Dictionary of metric histories
            save: Whether to save plots
        """
        # Separate train and val metrics
        train_metrics = {}
        val_metrics = {}
        
        for key, values in history.items():
            if key.startswith('train_'):
                metric_name = key.replace('train_', '')
                train_metrics[metric_name] = values
            elif key.startswith('val_'):
                metric_name = key.replace('val_', '')
                val_metrics[metric_name] = values
        
        # Get common metrics
        common_metrics = set(train_metrics.keys()) & set(val_metrics.keys())
        
        # Plot each metric
        for metric in common_metrics:
            self._plot_metric(
                metric,
                train_metrics[metric],
                val_metrics[metric],
                save=save
            )
        
        # Save history to CSV
        if save:
            self._save_history_csv(history)
    
    def _plot_metric(
        self,
        metric_name: str,
        train_values: List[float],
        val_values: List[float],
        save: bool = True
    ) -> None:
        """
        Plot single metric.
        
        Args:
            metric_name: Name of metric
            train_values: Training values
            val_values: Validation values
            save: Whether to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(len(train_values))
        
        # Plot lines
        ax.plot(epochs, train_values, marker='o', label='Train', linewidth=2)
        ax.plot(epochs, val_values, marker='s', label='Validation', linewidth=2)
        
        # Labels and title
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Over Epochs', fontsize=14, fontweight='bold')
        
        # Grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Set x-axis to integers
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
        # Add best values as text
        if metric_name == 'loss':
            best_train = min(train_values)
            best_val = min(val_values)
            best_label = 'Lowest'
        else:
            best_train = max(train_values)
            best_val = max(val_values)
            best_label = 'Highest'
        
        textstr = f'{best_label} Train: {best_train:.4f}\n{best_label} Val: {best_val:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f'{metric_name}_history.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _save_history_csv(self, history: Dict[str, List[float]]) -> None:
        """
        Save history to CSV.
        
        Args:
            history: Metrics history
        """
        # Create DataFrame
        max_len = max(len(v) for v in history.values())
        
        data = {}
        for key, values in history.items():
            # Pad if necessary
            padded = values + [None] * (max_len - len(values))
            data[key] = padded
        
        df = pd.DataFrame(data)
        df.insert(0, 'epoch', range(len(df)))
        
        # Save
        output_path = self.output_dir / 'training_history.csv'
        df.to_csv(output_path, index=False)
    
    def plot_confusion_matrix(
        self,
        cm,
        class_names: List[str],
        normalize: bool = False,
        save: bool = True
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize
            save: Whether to save plot
        """
        import seaborn as sns
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, None]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'accuracy',
        save: bool = True
    ) -> None:
        """
        Plot comparison across different models/levels.
        
        Args:
            results: Dictionary of {name: {metric: value}}
            metric: Metric to compare
            save: Whether to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(results.keys())
        values = [results[name][metric] for name in names]
        
        bars = ax.bar(names, values, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f'{metric}_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()