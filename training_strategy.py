"""
Training strategies for different hierarchical classification methods.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from pathlib import Path
import pandas as pd

from config import Config
from hierarchy import HierarchyManager
from data_module import DataModule
from trainer import Trainer
from logger import get_logger
from visualizer import Visualizer


class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""
    
    def __init__(
        self,
        config: Config,
        hierarchy: HierarchyManager,
        data_module: DataModule
    ):
        """
        Initialize training strategy.
        
        Args:
            config: Configuration object
            hierarchy: Hierarchy manager
            data_module: Data module
        """
        self.config = config
        self.hierarchy = hierarchy
        self.data_module = data_module
        self.logger = get_logger()
        self.visualizer = Visualizer(config.paths.log_dir / config.training.method)
    
    @abstractmethod
    def train(self) -> Dict[str, float]:
        """
        Execute training strategy.
        
        Returns:
            Dictionary of final test metrics
        """
        pass
    
    @abstractmethod
    def test(self) -> Dict[str, float]:
        """
        Execute testing.
        
        Returns:
            Dictionary of test metrics
        """
        pass
    
    def visualize_results(self, history: Dict) -> None:
        """
        Create visualization of training results.
        
        Args:
            history: Metrics history
        """
        self.visualizer.plot_training_history(history)


class FlatStrategy(TrainingStrategy):
    """Flat classification strategy (all leaf classes)."""
    
    def train(self) -> Dict[str, float]:
        """Train flat classifier."""
        self.logger.info("="*60)
        self.logger.info("FLAT CLASSIFICATION TRAINING")
        self.logger.info("="*60)
        
        # Get number of classes
        level_on_nodes, _, _, _ = self.hierarchy.get_hierarchy_info()
        max_level = max(level_on_nodes.keys())
        num_classes = len(level_on_nodes[max_level])
        
        self.logger.info(f"Number of leaf classes: {num_classes}")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.data_module.prepare_flat_data()
        self.test_loader = test_loader
        
        # Initialize trainer
        checkpoint_dir = self.config.paths.checkpoint_dir / 'flat'
        self.trainer = Trainer(
            config=self.config,
            num_classes=num_classes,
            checkpoint_dir=checkpoint_dir
        )
        
        # Train
        history = self.trainer.fit(train_loader, val_loader)
        
        # Visualize
        self.visualize_results(self.trainer.get_metrics_history())
        
        return history
    
    def test(self) -> Dict[str, float]:
        """Test flat classifier."""
        self.logger.info("="*60)
        self.logger.info("FLAT CLASSIFICATION TESTING")
        self.logger.info("="*60)
        
        # Test
        test_metrics = self.trainer.test(self.test_loader)
        
        # Log results
        self.logger.info("\nTest Results:")
        for key, value in test_metrics.items():
            if key == 'loss':
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value:.4f} ({value*100:.2f}%)")
        
        # Save results
        results_path = self.config.paths.log_dir / 'flat' / 'test_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([test_metrics]).to_csv(results_path, index=False)
        
        return test_metrics


class LevelStrategy(TrainingStrategy):
    """Level-wise classification strategy."""
    
    def train(self) -> Dict[str, float]:
        """Train level-wise classifiers."""
        self.logger.info("="*60)
        self.logger.info("LEVEL-WISE CLASSIFICATION TRAINING")
        self.logger.info("="*60)
        
        level_on_nodes, _, _, _ = self.hierarchy.get_hierarchy_info()
        num_levels = len(level_on_nodes)
        
        self.logger.info(f"Number of levels: {num_levels}")
        
        self.trainers = []
        all_histories = {}
        
        for level in range(num_levels):
            num_classes = len(level_on_nodes[level])
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training Level {level} ({num_classes} classes)")
            self.logger.info(f"{'='*60}")
            
            # Prepare data
            train_loader, val_loader, test_loader = \
                self.data_module.prepare_level_data(level)
            
            # Initialize trainer
            checkpoint_dir = self.config.paths.checkpoint_dir / 'level' / f'level_{level}'
            trainer = Trainer(
                config=self.config,
                num_classes=num_classes,
                checkpoint_dir=checkpoint_dir
            )
            
            # Transfer weights from previous level
            if level > 0 and self.trainers:
                self._transfer_weights(self.trainers[-1], trainer)
            
            # Train
            history = trainer.fit(train_loader, val_loader)
            
            # Store
            self.trainers.append(trainer)
            all_histories[f'level_{level}'] = trainer.get_metrics_history()
            
            # Visualize
            visualizer = Visualizer(
                self.config.paths.log_dir / 'level' / f'level_{level}'
            )
            visualizer.plot_training_history(trainer.get_metrics_history())
        
        return all_histories
    
    def _transfer_weights(self, source_trainer: Trainer, target_trainer: Trainer) -> None:
        """
        Transfer encoder weights from source to target trainer.
        
        Args:
            source_trainer: Source trainer with trained weights
            target_trainer: Target trainer to receive weights
        """
        self.logger.info("Transferring encoder weights from previous level...")
        
        # Get state dicts
        source_state = source_trainer.model.state_dict()
        target_state = target_trainer.model.state_dict()
        
        # Transfer matching weights (encoder layers)
        for key in source_state:
            if key.startswith('bert') or key.startswith('conv'):
                if key in target_state:
                    target_state[key] = source_state[key]
        
        target_trainer.model.load_state_dict(target_state)
        self.logger.info("Weight transfer completed")
    
    def test(self) -> Dict[str, float]:
        """Test level-wise classifiers."""
        self.logger.info("="*60)
        self.logger.info("LEVEL-WISE CLASSIFICATION TESTING")
        self.logger.info("="*60)
        
        all_results = {}
        
        for level, trainer in enumerate(self.trainers):
            self.logger.info(f"\nTesting Level {level}")
            
            # Prepare test data
            _, _, test_loader = self.data_module.prepare_level_data(level)
            
            # Test
            test_metrics = trainer.test(test_loader)
            
            # Log results
            self.logger.info(f"\nLevel {level} Test Results:")
            for key, value in test_metrics.items():
                if key == 'loss':
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value:.4f} ({value*100:.2f}%)")
            
            all_results[f'level_{level}'] = test_metrics
        
        # Save results
        results_path = self.config.paths.log_dir / 'level' / 'test_results.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).T.to_csv(results_path)
        
        return all_results


class SectionStrategy(TrainingStrategy):
    """Section-wise classification strategy."""
    
    def train(self) -> Dict[str, float]:
        """Train section-wise classifiers."""
        self.logger.info("="*60)
        self.logger.info("SECTION-WISE CLASSIFICATION TRAINING")
        self.logger.info("="*60)
        
        _, idx_on_section, _, _ = self.hierarchy.get_hierarchy_info()
        sections = list(idx_on_section.keys())
        
        self.logger.info(f"Number of sections: {len(sections)}")
        
        self.trainers = {}
        all_histories = {}
        
        for section in sections:
            num_classes = len(idx_on_section[section])
            
            # Skip sections with only one class
            if num_classes <= 1:
                self.logger.info(f"Skipping section {section} (only 1 class)")
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training Section {section} ({num_classes} classes)")
            self.logger.info(f"{'='*60}")
            
            # Prepare data
            train_loader, val_loader, test_loader = \
                self.data_module.prepare_section_data(section)
            
            # Initialize trainer
            checkpoint_dir = self.config.paths.checkpoint_dir / 'section' / f'section_{section}'
            trainer = Trainer(
                config=self.config,
                num_classes=num_classes,
                checkpoint_dir=checkpoint_dir
            )
            
            # Train
            history = trainer.fit(train_loader, val_loader)
            
            # Store
            self.trainers[section] = trainer
            all_histories[f'section_{section}'] = trainer.get_metrics_history()
            
            # Visualize
            visualizer = Visualizer(
                self.config.paths.log_dir / 'section' / f'section_{section}'
            )
            visualizer.plot_training_history(trainer.get_metrics_history())
        
        return all_histories
    
    def test(self) -> Dict[str, float]:
        """Test section-wise classifiers (hierarchical inference)."""
        self.logger.info("="*60)
        self.logger.info("SECTION-WISE CLASSIFICATION TESTING")
        self.logger.info("="*60)
        
        # TODO: Implement hierarchical testing
        # This requires navigating the tree structure during inference
        
        self.logger.info("Hierarchical testing not yet implemented")
        return {}


def create_strategy(
    method: str,
    config: Config,
    hierarchy: HierarchyManager,
    data_module: DataModule
) -> TrainingStrategy:
    """
    Factory function to create training strategy.
    
    Args:
        method: Training method ('flat', 'level', or 'section')
        config: Configuration object
        hierarchy: Hierarchy manager
        data_module: Data module
        
    Returns:
        Training strategy instance
    """
    strategies = {
        'flat': FlatStrategy,
        'level': LevelStrategy,
        'section': SectionStrategy
    }
    
    if method not in strategies:
        raise ValueError(f"Unknown method: {method}. Choose from {list(strategies.keys())}")
    
    return strategies[method](config, hierarchy, data_module)