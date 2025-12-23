"""
Unified trainer for all classification methods.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np

from config import Config
from model import create_model
from metrics import MetricsCalculator, MetricsTracker
from logger import get_logger


class Trainer:
    """Unified trainer for hierarchical text classification."""
    
    def __init__(
        self,
        config: Config,
        num_classes: int,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            num_classes: Number of output classes
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
        """
        self.config = config
        self.num_classes = num_classes
        self.logger = get_logger()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize model
        self.model = create_model(
            num_classes=num_classes,
            config=config,
            use_output_layer=True
        )
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=0.5,
            total_iters=config.training.warmup_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.metrics_calculator = MetricsCalculator(
            num_classes=num_classes,
            device=self.device
        )
        self.metrics_tracker = MetricsTracker()
        
        # Checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = config.paths.checkpoint_dir / config.training.method
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.patience = config.training.patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.training.seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.metrics_calculator.reset()
        
        epoch_loss = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids, targets = batch
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip_val
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Compute metrics
            preds = torch.argmax(logits, dim=1)
            batch_metrics = self.metrics_calculator.compute(preds, targets)
            batch_metrics['loss'] = loss.item()
            
            epoch_loss.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_metrics['accuracy']:.4f}"
            })
            
            # Track metrics
            self.metrics_tracker.update(batch_metrics, prefix='train_')
        
        # End epoch
        epoch_metrics = self.metrics_tracker.end_epoch()
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        epoch_loss = []
        
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            input_ids, targets = batch
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            loss = self.criterion(logits, targets)
            
            # Compute metrics
            preds = torch.argmax(logits, dim=1)
            batch_metrics = self.metrics_calculator.compute(preds, targets)
            batch_metrics['loss'] = loss.item()
            
            epoch_loss.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_metrics['accuracy']:.4f}"
            })
            
            # Track metrics
            self.metrics_tracker.update(batch_metrics, prefix='val_')
        
        # End epoch
        epoch_metrics = self.metrics_tracker.end_epoch()
        
        return epoch_metrics
    
    @torch.no_grad()
    def test(self, test_loader) -> Dict[str, float]:
        """
        Test model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        all_preds = []
        all_targets = []
        epoch_loss = []
        
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for batch in progress_bar:
            input_ids, targets = batch
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            loss = self.criterion(logits, targets)
            
            # Compute predictions
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            epoch_loss.append(loss.item())
        
        # Compute final metrics
        all_preds = torch.tensor(all_preds).to(self.device)
        all_targets = torch.tensor(all_targets).to(self.device)
        
        test_metrics = self.metrics_calculator.compute(all_preds, all_targets)
        test_metrics['loss'] = np.mean(epoch_loss)
        
        return test_metrics
    
    def fit(
        self,
        train_loader,
        val_loader,
        max_epochs: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_epochs: Maximum number of epochs (overrides config)
            
        Returns:
            Best validation metrics
        """
        if max_epochs is None:
            max_epochs = self.config.training.max_epochs
        
        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(max_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Print summary
            self.metrics_tracker.print_epoch_summary(epoch)
            
            # Check for improvement
            val_loss = val_metrics['val_loss']
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics)
                
                self.logger.info(f"New best model at epoch {epoch} with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                self.logger.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Load best checkpoint
        self.load_checkpoint()
        
        self.logger.info(f"Training completed. Best epoch: {self.best_epoch}")
        
        return self.metrics_tracker.get_history('val_loss')
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (uses best if None)
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
    
    def get_metrics_history(self) -> Dict[str, list]:
        """Get full metrics history."""
        return self.metrics_tracker.history