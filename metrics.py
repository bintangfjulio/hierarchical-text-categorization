"""
Metrics computation and tracking.
"""
import torch
from typing import Dict, List, Optional
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall
)


class MetricsCalculator:
    """Calculate and track classification metrics."""
    
    def __init__(self, num_classes: int, device: torch.device):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            device: Device to use for computation
        """
        self.num_classes = num_classes
        self.device = device
        
        # Initialize metrics
        self.accuracy = MulticlassAccuracy(
            num_classes=num_classes
        ).to(device)
        
        self.f1_micro = MulticlassF1Score(
            num_classes=num_classes,
            average='micro'
        ).to(device)
        
        self.f1_macro = MulticlassF1Score(
            num_classes=num_classes,
            average='macro'
        ).to(device)
        
        self.f1_weighted = MulticlassF1Score(
            num_classes=num_classes,
            average='weighted'
        ).to(device)
        
        self.precision = MulticlassPrecision(
            num_classes=num_classes,
            average='macro'
        ).to(device)
        
        self.recall = MulticlassRecall(
            num_classes=num_classes,
            average='macro'
        ).to(device)
    
    def compute(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            preds: Predicted class indices
            targets: Ground truth class indices
            
        Returns:
            Dictionary of metric name to value
        """
        metrics = {
            'accuracy': self.accuracy(preds, targets).item(),
            'f1_micro': self.f1_micro(preds, targets).item(),
            'f1_macro': self.f1_macro(preds, targets).item(),
            'f1_weighted': self.f1_weighted(preds, targets).item(),
            'precision': self.precision(preds, targets).item(),
            'recall': self.recall(preds, targets).item()
        }
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.accuracy.reset()
        self.f1_micro.reset()
        self.f1_macro.reset()
        self.f1_weighted.reset()
        self.precision.reset()
        self.recall.reset()


class MetricsTracker:
    """Track metrics across epochs and batches."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history: Dict[str, List[float]] = {}
        self.current_epoch: Dict[str, List[float]] = {}
    
    def update(self, metrics: Dict[str, float], prefix: str = '') -> None:
        """
        Update metrics for current batch.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for metric names (e.g., 'train_', 'val_')
        """
        for key, value in metrics.items():
            metric_name = f"{prefix}{key}"
            
            if metric_name not in self.current_epoch:
                self.current_epoch[metric_name] = []
            
            self.current_epoch[metric_name].append(value)
    
    def end_epoch(self) -> Dict[str, float]:
        """
        Compute epoch averages and save to history.
        
        Returns:
            Dictionary of averaged metrics for the epoch
        """
        epoch_metrics = {}
        
        for metric_name, values in self.current_epoch.items():
            avg_value = sum(values) / len(values) if values else 0.0
            epoch_metrics[metric_name] = avg_value
            
            if metric_name not in self.history:
                self.history[metric_name] = []
            
            self.history[metric_name].append(avg_value)
        
        self.current_epoch.clear()
        
        return epoch_metrics
    
    def get_best(self, metric_name: str, mode: str = 'max') -> float:
        """
        Get best value for a metric.
        
        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'
            
        Returns:
            Best metric value
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return float('-inf') if mode == 'max' else float('inf')
        
        values = self.history[metric_name]
        return max(values) if mode == 'max' else min(values)
    
    def get_history(self, metric_name: str) -> List[float]:
        """
        Get full history of a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            List of metric values across epochs
        """
        return self.history.get(metric_name, [])
    
    def get_last(self, metric_name: str) -> Optional[float]:
        """
        Get last value of a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Last metric value or None
        """
        history = self.history.get(metric_name, [])
        return history[-1] if history else None
    
    def print_epoch_summary(self, epoch: int) -> None:
        """
        Print summary of metrics for current epoch.
        
        Args:
            epoch: Epoch number
        """
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*60}")
        
        # Group metrics by prefix
        train_metrics = {}
        val_metrics = {}
        
        for key in self.history:
            if not self.history[key]:
                continue
            
            value = self.history[key][-1]
            
            if key.startswith('train_'):
                train_metrics[key.replace('train_', '')] = value
            elif key.startswith('val_'):
                val_metrics[key.replace('val_', '')] = value
        
        # Print training metrics
        if train_metrics:
            print("\nTraining Metrics:")
            for key, value in sorted(train_metrics.items()):
                if key == 'loss':
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value:.4f} ({value*100:.2f}%)")
        
        # Print validation metrics
        if val_metrics:
            print("\nValidation Metrics:")
            for key, value in sorted(val_metrics.items()):
                if key == 'loss':
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value:.4f} ({value*100:.2f}%)")
        
        print(f"{'='*60}\n")