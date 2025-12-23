"""
Inference script for hierarchical text classification.

Usage:
    python inference.py --method flat --text "smartphone android murah"
    python inference.py --method section --text "sepatu olahraga nike" --verbose
"""
import argparse
import torch
from pathlib import Path
from typing import List, Tuple

from config import Config
from hierarchy import HierarchyManager
from text_processor import TextProcessor
from model import BERTCNN
from transformers import BertTokenizer
from logger import get_logger


class Predictor:
    """Predictor for hierarchical text classification."""
    
    def __init__(
        self,
        config: Config,
        hierarchy: HierarchyManager,
        checkpoint_dir: Path,
        device: torch.device = None
    ):
        """
        Initialize predictor.
        
        Args:
            config: Configuration object
            hierarchy: Hierarchy manager
            checkpoint_dir: Directory containing checkpoints
            device: Device to use for inference
        """
        self.config = config
        self.hierarchy = hierarchy
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = get_logger()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize text processor and tokenizer
        self.text_processor = TextProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(config.model.bert_model)
        
        self.hierarchy.load_hierarchy()
        
        self.logger.info(f"Predictor initialized on {self.device}")
    
    def predict_flat(self, text: str, verbose: bool = False) -> Tuple[str, float]:
        """
        Predict using flat classification.
        
        Args:
            text: Input text
            verbose: Whether to print detailed info
            
        Returns:
            Tuple of (predicted_category, confidence)
        """
        # Get leaf classes
        level_on_nodes, _, _, _ = self.hierarchy.get_hierarchy_info()
        max_level = max(level_on_nodes.keys())
        leaf_classes = level_on_nodes[max_level]
        num_classes = len(leaf_classes)
        
        # Load model
        checkpoint_path = self.checkpoint_dir / 'flat' / 'best_model.pt'
        model = self._load_model(checkpoint_path, num_classes)
        
        # Preprocess
        processed_text = self.text_processor.clean_text(text)
        
        # Tokenize
        encoded = self.tokenizer(
            processed_text,
            max_length=self.config.data.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(input_ids)
            probabilities = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)
        
        predicted_category = leaf_classes[pred_idx.item()]
        confidence_score = confidence.item()
        
        if verbose:
            self.logger.info(f"Input: {text}")
            self.logger.info(f"Cleaned: {processed_text}")
            self.logger.info(f"Predicted: {predicted_category}")
            self.logger.info(f"Confidence: {confidence_score:.4f}")
        
        return predicted_category, confidence_score
    
    def predict_section(
        self,
        text: str,
        verbose: bool = False
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Predict using hierarchical section-wise classification.
        
        Args:
            text: Input text
            verbose: Whether to print detailed info
            
        Returns:
            Tuple of (final_category, path_with_confidences)
        """
        level_on_nodes, idx_on_section, section_on_idx, section_parent_child = \
            self.hierarchy.get_hierarchy_info()
        
        num_levels = len(level_on_nodes)
        
        # Preprocess
        processed_text = self.text_processor.clean_text(text)
        
        # Tokenize
        encoded = self.tokenizer(
            processed_text,
            max_length=self.config.data.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        
        # Navigate hierarchy
        path_with_confidences = []
        
        # Start from root
        current_section = section_on_idx[list(section_parent_child['root'])[0]]
        
        if verbose:
            self.logger.info(f"Input: {text}")
            self.logger.info(f"Cleaned: {processed_text}")
            self.logger.info("\nNavigating hierarchy:")
        
        for level in range(num_levels):
            section_classes = idx_on_section[current_section]
            
            # Skip if only one class
            if len(section_classes) == 1:
                predicted_category = section_classes[0]
                confidence = 1.0
                
                if verbose:
                    self.logger.info(f"Level {level}: {predicted_category} (only option)")
            else:
                # Load model for this section
                checkpoint_path = self.checkpoint_dir / 'section' / f'section_{current_section}' / 'best_model.pt'
                
                if not checkpoint_path.exists():
                    self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
                    break
                
                model = self._load_model(checkpoint_path, len(section_classes))
                
                # Predict
                model.eval()
                with torch.no_grad():
                    logits = model(input_ids)
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, pred_idx = torch.max(probabilities, dim=1)
                
                predicted_category = section_classes[pred_idx.item()]
                confidence = confidence.item()
                
                if verbose:
                    self.logger.info(f"Level {level}: {predicted_category} (confidence: {confidence:.4f})")
            
            path_with_confidences.append((predicted_category, confidence))
            
            # Move to next level
            if level < num_levels - 1:
                if predicted_category in section_parent_child:
                    children = list(section_parent_child[predicted_category])
                    if children:
                        next_node = children[0]
                        current_section = section_on_idx[next_node]
                    else:
                        break
                else:
                    break
        
        final_category = " > ".join([cat for cat, _ in path_with_confidences])
        
        if verbose:
            self.logger.info(f"\nFinal prediction: {final_category}")
        
        return final_category, path_with_confidences
    
    def _load_model(self, checkpoint_path: Path, num_classes: int) -> BERTCNN:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            num_classes: Number of output classes
            
        Returns:
            Loaded model
        """
        model = BERTCNN(
            num_classes=num_classes,
            bert_model=self.config.model.bert_model,
            dropout=self.config.model.dropout,
            window_sizes=self.config.model.window_sizes,
            out_channels=self.config.model.out_channels
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Inference for Hierarchical Text Classification'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['flat', 'section'],
        help='Classification method'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        required=True,
        help='Input text to classify'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='large',
        help='Dataset name'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Checkpoint directory'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_arguments()
    
    # Load config
    config_path = Path('logs') / args.method / f'{args.method}_config.json'
    
    if config_path.exists():
        config = Config.from_json(config_path)
    else:
        # Use default config
        config = Config()
        config.data.dataset = args.dataset
    
    # Initialize hierarchy
    tree_file = Path('datasets') / f'{config.data.dataset}_hierarchy.tree'
    hierarchy = HierarchyManager(tree_file)
    
    # Initialize predictor
    predictor = Predictor(
        config=config,
        hierarchy=hierarchy,
        checkpoint_dir=Path(args.checkpoint_dir)
    )
    
    # Predict
    if args.method == 'flat':
        category, confidence = predictor.predict_flat(args.text, verbose=args.verbose)
        print(f"\nPredicted Category: {category}")
        print(f"Confidence: {confidence:.4f}")
    
    elif args.method == 'section':
        final_category, path = predictor.predict_section(args.text, verbose=args.verbose)
        print(f"\nFull Path: {final_category}")
        print("\nHierarchy:")
        for i, (cat, conf) in enumerate(path):
            print(f"  Level {i}: {cat} (confidence: {conf:.4f})")


if __name__ == '__main__':
    main()