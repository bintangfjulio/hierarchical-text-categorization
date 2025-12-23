"""
PyTorch dataset classes for hierarchical classification.
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import pandas as pd


class FlatDataset(Dataset):
    """Dataset for flat classification (all classes at leaf level)."""
    
    def __init__(
        self,
        input_ids: List[List[int]],
        targets: List[int]
    ):
        """
        Initialize flat dataset.
        
        Args:
            input_ids: List of tokenized input sequences
            targets: List of target class indices
        """
        assert len(input_ids) == len(targets), \
            f"Mismatched lengths: {len(input_ids)} vs {len(targets)}"
        
        self.input_ids = input_ids
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


class LevelDataset(Dataset):
    """Dataset for level-wise classification."""
    
    def __init__(
        self,
        input_ids: List[List[int]],
        targets: List[int],
        level: int
    ):
        """
        Initialize level dataset.
        
        Args:
            input_ids: List of tokenized input sequences
            targets: List of target class indices for this level
            level: Current level index
        """
        assert len(input_ids) == len(targets), \
            f"Mismatched lengths: {len(input_ids)} vs {len(targets)}"
        
        self.input_ids = input_ids
        self.targets = targets
        self.level = level
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


class SectionDataset(Dataset):
    """Dataset for section-wise classification."""
    
    def __init__(
        self,
        input_ids: List[List[int]],
        targets: List[int],
        section: int
    ):
        """
        Initialize section dataset.
        
        Args:
            input_ids: List of tokenized input sequences
            targets: List of target class indices for this section
            section: Current section index
        """
        assert len(input_ids) == len(targets), \
            f"Mismatched lengths: {len(input_ids)} vs {len(targets)}"
        
        self.input_ids = input_ids
        self.targets = targets
        self.section = section
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


class HierarchicalDataset(Dataset):
    """Dataset for hierarchical classification with full path."""
    
    def __init__(
        self,
        input_ids: List[List[int]],
        targets_per_level: List[List[int]],
        hierarchy_paths: Optional[List[str]] = None
    ):
        """
        Initialize hierarchical dataset.
        
        Args:
            input_ids: List of tokenized input sequences
            targets_per_level: List of target lists, one per hierarchy level
            hierarchy_paths: Optional list of full hierarchy paths
        """
        self.input_ids = input_ids
        self.targets_per_level = targets_per_level
        self.hierarchy_paths = hierarchy_paths
        self.num_levels = len(targets_per_level[0]) if targets_per_level else 0
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Get item with input and all level targets.
        
        Returns:
            Tuple of (input_ids, list of targets per level)
        """
        input_tensor = torch.tensor(self.input_ids[idx], dtype=torch.long)
        target_tensors = [
            torch.tensor(level_targets[idx], dtype=torch.long)
            for level_targets in self.targets_per_level
        ]
        
        return input_tensor, target_tensors


class InferenceDataset(Dataset):
    """Dataset for inference/prediction."""
    
    def __init__(
        self,
        input_ids: List[List[int]],
        original_texts: Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None
    ):
        """
        Initialize inference dataset.
        
        Args:
            input_ids: List of tokenized input sequences
            original_texts: Optional list of original text strings
            metadata: Optional list of metadata dictionaries
        """
        self.input_ids = input_ids
        self.original_texts = original_texts or [None] * len(input_ids)
        self.metadata = metadata or [{} for _ in range(len(input_ids))]
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Dict]:
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.long),
            self.original_texts[idx],
            self.metadata[idx]
        )


def collate_hierarchical(batch: List[Tuple]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Custom collate function for hierarchical dataset.
    
    Args:
        batch: List of samples from HierarchicalDataset
        
    Returns:
        Batched tensors
    """
    input_ids = torch.stack([item[0] for item in batch])
    
    # Stack targets for each level
    num_levels = len(batch[0][1])
    targets_per_level = [
        torch.stack([item[1][level] for item in batch])
        for level in range(num_levels)
    ]
    
    return input_ids, targets_per_level