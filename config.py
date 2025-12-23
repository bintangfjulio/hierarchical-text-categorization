"""
Configuration management for hierarchical text classification.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    bert_model: str = 'indolem/indobert-base-uncased'
    dropout: float = 0.1
    window_sizes: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    in_channels: int = 4
    out_channels: int = 32
    input_size: int = 768


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    method: str = 'flat'  # 'flat', 'level', or 'section'
    batch_size: int = 32
    max_epochs: int = 50
    learning_rate: float = 2e-5
    patience: int = 3
    seed: int = 42
    gradient_clip_val: float = 1.0
    warmup_steps: int = 500
    weight_decay: float = 0.01


@dataclass
class DataConfig:
    """Data processing configuration."""
    dataset: str = 'large'
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    max_length: int = 512
    num_workers: int = 4
    cache_dir: Path = field(default_factory=lambda: Path('cache'))


@dataclass
class PathConfig:
    """File paths configuration."""
    root_dir: Path = field(default_factory=lambda: Path('.'))
    data_dir: Path = field(default_factory=lambda: Path('datasets'))
    checkpoint_dir: Path = field(default_factory=lambda: Path('checkpoints'))
    log_dir: Path = field(default_factory=lambda: Path('logs'))
    cache_dir: Path = field(default_factory=lambda: Path('cache'))
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data_dir, self.checkpoint_dir, self.log_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration class combining all configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            paths=PathConfig(**config_dict.get('paths', {}))
        )
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'Config':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'paths': {k: str(v) for k, v in self.paths.__dict__.items()}
        }
    
    def save(self, json_path: Path) -> None:
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> None:
        """Validate configuration values."""
        assert self.training.method in ['flat', 'level', 'section'], \
            f"Invalid method: {self.training.method}"
        assert 0 < self.training.learning_rate < 1, \
            f"Invalid learning rate: {self.training.learning_rate}"
        assert 0 <= self.model.dropout < 1, \
            f"Invalid dropout: {self.model.dropout}"
        assert self.training.batch_size > 0, \
            f"Invalid batch size: {self.training.batch_size}"
        assert self.data.train_ratio + self.data.valid_ratio + self.data.test_ratio == 1.0, \
            "Data split ratios must sum to 1.0"