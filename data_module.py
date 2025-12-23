"""
Data module for loading and preprocessing datasets.
"""
import os
import pickle
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from tqdm import tqdm

from config import Config
from hierarchy import HierarchyManager
from text_processor import TextProcessor, TokenizerWrapper
from dataset import FlatDataset, LevelDataset, SectionDataset
from logger import get_logger


class DataModule:
    """Data module for hierarchical text classification."""
    
    def __init__(
        self,
        config: Config,
        hierarchy: HierarchyManager
    ):
        """
        Initialize data module.
        
        Args:
            config: Configuration object
            hierarchy: Hierarchy manager
        """
        self.config = config
        self.hierarchy = hierarchy
        self.logger = get_logger()
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(config.model.bert_model)
        self.tokenizer_wrapper = TokenizerWrapper(
            self.tokenizer,
            max_length=config.data.max_length
        )
        
        # Dataset paths
        self.dataset_path = self._get_dataset_path()
        self.cache_dir = config.paths.cache_dir / config.data.dataset
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load raw dataset
        self.raw_data = self._load_raw_dataset()
        self.max_length = self._calculate_max_length()
        
        # Processed datasets
        self.train_data: Optional[pd.DataFrame] = None
        self.valid_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
    
    def _get_dataset_path(self) -> Path:
        """Get or download dataset."""
        dataset_name = self.config.data.dataset
        dataset_file = self.config.paths.data_dir / f'{dataset_name}_product_tokopedia.csv'
        
        if not dataset_file.exists():
            self.logger.info(f"Downloading dataset: {dataset_name}")
            url = f'https://github.com/bintangfjulio/product_categories_classification/releases/download/{dataset_name}/{dataset_name}_product_tokopedia.csv'
            
            try:
                response = requests.get(url, allow_redirects=True, timeout=60)
                response.raise_for_status()
                
                with open(dataset_file, 'wb') as f:
                    f.write(response.content)
                
                self.logger.info(f"Dataset downloaded: {dataset_file}")
            except Exception as e:
                self.logger.error(f"Failed to download dataset: {e}")
                raise
        
        return dataset_file
    
    def _load_raw_dataset(self) -> pd.DataFrame:
        """Load raw dataset from CSV."""
        self.logger.info(f"Loading dataset from: {self.dataset_path}")
        
        try:
            df = pd.read_csv(self.dataset_path)
            self.logger.info(f"Loaded {len(df)} samples")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _calculate_max_length(self) -> int:
        """Calculate maximum sequence length."""
        cache_file = self.cache_dir / 'max_length.pkl'
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                max_length = pickle.load(f)
            self.logger.info(f"Loaded cached max_length: {max_length}")
            return max_length
        
        self.logger.info("Calculating maximum sequence length...")
        texts = self.raw_data.iloc[:, 0].astype(str).tolist()
        max_length = self.text_processor.get_max_length(texts)
        max_length = min(max_length, self.config.data.max_length)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(max_length, f)
        
        self.logger.info(f"Maximum sequence length: {max_length}")
        return max_length
    
    def split_data(self) -> None:
        """Split data into train/valid/test sets."""
        cache_file = self.cache_dir / 'data_splits.pkl'
        
        if cache_file.exists():
            self.logger.info("Loading cached data splits...")
            with open(cache_file, 'rb') as f:
                splits = pickle.load(f)
            
            self.train_data = splits['train']
            self.valid_data = splits['valid']
            self.test_data = splits['test']
            
            self.logger.info(f"Train: {len(self.train_data)}, Valid: {len(self.valid_data)}, Test: {len(self.test_data)}")
            return
        
        self.logger.info("Splitting dataset...")
        
        # Shuffle data
        data = self.raw_data.sample(frac=1, random_state=self.config.training.seed).reset_index(drop=True)
        
        # Calculate split sizes
        n = len(data)
        train_size = int(n * self.config.data.train_ratio)
        valid_size = int(n * self.config.data.valid_ratio)
        
        # Split
        self.train_data = data.iloc[:train_size].reset_index(drop=True)
        self.valid_data = data.iloc[train_size:train_size + valid_size].reset_index(drop=True)
        self.test_data = data.iloc[train_size + valid_size:].reset_index(drop=True)
        
        # Cache splits
        splits = {
            'train': self.train_data,
            'valid': self.valid_data,
            'test': self.test_data
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(splits, f)
        
        self.logger.info(f"Train: {len(self.train_data)}, Valid: {len(self.valid_data)}, Test: {len(self.test_data)}")
    
    def prepare_flat_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare dataloaders for flat classification."""
        cache_file = self.cache_dir / 'flat_processed.pkl'
        
        if cache_file.exists():
            self.logger.info("Loading cached flat data...")
            with open(cache_file, 'rb') as f:
                processed = pickle.load(f)
        else:
            self.logger.info("Processing flat data...")
            
            if self.train_data is None:
                self.split_data()
            
            processed = {}
            for split_name, split_data in [
                ('train', self.train_data),
                ('valid', self.valid_data),
                ('test', self.test_data)
            ]:
                input_ids, targets = self._process_flat_split(split_data)
                processed[split_name] = (input_ids, targets)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(processed, f)
        
        # Create datasets
        train_dataset = FlatDataset(*processed['train'])
        valid_dataset = FlatDataset(*processed['valid'])
        test_dataset = FlatDataset(*processed['test'])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, valid_loader, test_loader
    
    def _process_flat_split(self, data: pd.DataFrame) -> Tuple[List, List]:
        """Process data split for flat classification."""
        level_on_nodes, _, _, _ = self.hierarchy.get_hierarchy_info()
        max_level = max(level_on_nodes.keys())
        leaf_classes = level_on_nodes[max_level]
        
        input_ids = []
        targets = []
        
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing"):
            # Clean text
            text = self.text_processor.clean_text(str(row.iloc[0]))
            
            # Tokenize
            encoded = self.tokenizer_wrapper.encode(text)
            input_ids.append(encoded['input_ids'])
            
            # Extract target (last node in path)
            hierarchy_path = str(row.iloc[-1])
            last_node = hierarchy_path.split(" > ")[-1].lower()
            target_idx = leaf_classes.index(last_node)
            targets.append(target_idx)
        
        return input_ids, targets
    
    def prepare_level_data(self, level: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare dataloaders for level-wise classification."""
        cache_file = self.cache_dir / f'level_{level}_processed.pkl'
        
        if cache_file.exists():
            self.logger.info(f"Loading cached level {level} data...")
            with open(cache_file, 'rb') as f:
                processed = pickle.load(f)
        else:
            self.logger.info(f"Processing level {level} data...")
            
            if self.train_data is None:
                self.split_data()
            
            processed = {}
            for split_name, split_data in [
                ('train', self.train_data),
                ('valid', self.valid_data),
                ('test', self.test_data)
            ]:
                input_ids, targets = self._process_level_split(split_data, level)
                processed[split_name] = (input_ids, targets)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(processed, f)
        
        # Create datasets
        train_dataset = LevelDataset(*processed['train'], level=level)
        valid_dataset = LevelDataset(*processed['valid'], level=level)
        test_dataset = LevelDataset(*processed['test'], level=level)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, valid_loader, test_loader
    
    def _process_level_split(self, data: pd.DataFrame, level: int) -> Tuple[List, List]:
        """Process data split for level-wise classification."""
        level_on_nodes, _, _, _ = self.hierarchy.get_hierarchy_info()
        level_classes = level_on_nodes[level]
        
        input_ids = []
        targets = []
        
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing level {level}"):
            # Clean text
            text = self.text_processor.clean_text(str(row.iloc[0]))
            
            # Tokenize
            encoded = self.tokenizer_wrapper.encode(text)
            input_ids.append(encoded['input_ids'])
            
            # Extract target at this level
            hierarchy_path = str(row.iloc[-1])
            nodes = hierarchy_path.split(" > ")
            
            if level < len(nodes):
                node_at_level = nodes[level].lower()
                target_idx = level_classes.index(node_at_level)
                targets.append(target_idx)
            else:
                # Handle edge case
                self.logger.warning(f"Skipping sample with incomplete path at level {level}")
        
        return input_ids, targets
    
    def prepare_section_data(self, section: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare dataloaders for section-wise classification."""
        cache_file = self.cache_dir / f'section_{section}_processed.pkl'
        
        if cache_file.exists():
            self.logger.info(f"Loading cached section {section} data...")
            with open(cache_file, 'rb') as f:
                processed = pickle.load(f)
        else:
            self.logger.info(f"Processing section {section} data...")
            
            if self.train_data is None:
                self.split_data()
            
            processed = {}
            for split_name, split_data in [
                ('train', self.train_data),
                ('valid', self.valid_data),
                ('test', self.test_data)
            ]:
                input_ids, targets = self._process_section_split(split_data, section)
                processed[split_name] = (input_ids, targets)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(processed, f)
        
        # Create datasets
        train_dataset = SectionDataset(*processed['train'], section=section)
        valid_dataset = SectionDataset(*processed['valid'], section=section)
        test_dataset = SectionDataset(*processed['test'], section=section)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, valid_loader, test_loader
    
    def _process_section_split(self, data: pd.DataFrame, section: int) -> Tuple[List, List]:
        """Process data split for section-wise classification."""
        _, idx_on_section, section_on_idx, _ = self.hierarchy.get_hierarchy_info()
        section_classes = idx_on_section[section]
        
        input_ids = []
        targets = []
        
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing section {section}"):
            hierarchy_path = str(row.iloc[-1])
            nodes = [node.lower() for node in hierarchy_path.split(" > ")]
            
            # Check if any node in path belongs to this section
            found = False
            for node in nodes:
                if node in section_classes:
                    # Clean text
                    text = self.text_processor.clean_text(str(row.iloc[0]))
                    
                    # Tokenize
                    encoded = self.tokenizer_wrapper.encode(text)
                    input_ids.append(encoded['input_ids'])
                    
                    # Target
                    target_idx = section_classes.index(node)
                    targets.append(target_idx)
                    
                    found = True
                    break
        
        return input_ids, targets