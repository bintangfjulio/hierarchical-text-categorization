"""
Main training script for hierarchical text classification.

Usage:
    python main.py --method flat --dataset large
    python main.py --method level --dataset large --max_epochs 30
    python main.py --method section --dataset small
"""
import argparse
import sys
from pathlib import Path

from config import Config, ModelConfig, TrainingConfig, DataConfig, PathConfig
from hierarchy import HierarchyManager
from data_module import DataModule
from training_strategy import create_strategy
from logger import get_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hierarchical Text Classification for Indonesian E-commerce Products'
    )
    
    # Method
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['flat', 'level', 'section'],
        help='Training method'
    )
    
    # Data
    parser.add_argument(
        '--dataset',
        type=str,
        default='large',
        help='Dataset name (e.g., large, small)'
    )
    
    # Model
    parser.add_argument(
        '--bert_model',
        type=str,
        default='indolem/indobert-base-uncased',
        help='Pre-trained BERT model'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate'
    )
    
    # Training
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help='Maximum number of epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=3,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Paths
    parser.add_argument(
        '--data_dir',
        type=str,
        default='datasets',
        help='Data directory'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Log directory'
    )
    
    # Actions
    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Only run testing (requires trained model)'
    )
    parser.add_argument(
        '--visualize_only',
        action='store_true',
        help='Only create visualizations'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Configuration object
    """
    model_config = ModelConfig(
        bert_model=args.bert_model,
        dropout=args.dropout
    )
    
    training_config = TrainingConfig(
        method=args.method,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed
    )
    
    data_config = DataConfig(
        dataset=args.dataset
    )
    
    path_config = PathConfig(
        data_dir=Path(args.data_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        log_dir=Path(args.log_dir)
    )
    
    config = Config(
        model=model_config,
        training=training_config,
        data=data_config,
        paths=path_config
    )
    
    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create config
    config = create_config_from_args(args)
    
    # Validate config
    try:
        config.validate()
    except AssertionError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Initialize logger
    logger = get_logger(log_dir=config.paths.log_dir)
    
    logger.info("="*60)
    logger.info("HIERARCHICAL TEXT CLASSIFICATION")
    logger.info("="*60)
    logger.info(f"Method: {config.training.method}")
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"BERT Model: {config.model.bert_model}")
    logger.info(f"Batch Size: {config.training.batch_size}")
    logger.info(f"Max Epochs: {config.training.max_epochs}")
    logger.info(f"Learning Rate: {config.training.learning_rate}")
    logger.info("="*60)
    
    # Save config
    config_path = config.paths.log_dir / f'{config.training.method}_config.json'
    config.save(config_path)
    logger.info(f"Configuration saved: {config_path}")
    
    try:
        # Initialize hierarchy
        tree_file = config.paths.data_dir / f'{config.data.dataset}_hierarchy.tree'
        
        if not tree_file.exists():
            logger.info("Creating hierarchy tree file from dataset...")
            # Will be created by data module
            hierarchy = HierarchyManager(tree_file)
        else:
            hierarchy = HierarchyManager(tree_file)
            hierarchy.load_hierarchy()
            hierarchy.print_hierarchy()
        
        # Initialize data module
        logger.info("Initializing data module...")
        data_module = DataModule(config, hierarchy)
        
        # Create training strategy
        logger.info(f"Creating {config.training.method} training strategy...")
        strategy = create_strategy(
            method=config.training.method,
            config=config,
            hierarchy=hierarchy,
            data_module=data_module
        )
        
        # Execute training/testing
        if args.test_only:
            logger.info("Running testing only...")
            test_results = strategy.test()
        elif args.visualize_only:
            logger.info("Creating visualizations only...")
            # Load history and visualize
            # This would need to be implemented
            pass
        else:
            # Full training and testing
            logger.info("Starting training...")
            train_results = strategy.train()
            
            logger.info("Starting testing...")
            test_results = strategy.test()
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()