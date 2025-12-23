# Fine-tune Deep Learning for Short Text Hierarchical Extra Large Categorization

A implementation of hierarchical text classification for Indonesian product categories using BERT-CNN architecture.

## Features

- **Training Methods**:
  - **Flat**: Single classifier for all leaf categories
  - **Level**: Separate classifiers per hierarchy level with weight transfer
  - **Section**: Separate classifiers per hierarchy section

- **Architecture**:
  - IndoBERT encoder
  - Multi-scale CNN for feature extraction
  - Hierarchical classification support

- **Experiment-Ready**:
  - Comprehensive configuration management
  - Proper logging and error handling
  - Automatic checkpointing and early stopping
  - Visualization tools
  - Extensive documentation

## Project Structure

```
.
├── config.py              # Configuration management
├── logger.py              # Centralized logging
├── hierarchy.py           # Hierarchy structure management
├── text_processor.py      # Text preprocessing
├── dataset.py             # PyTorch dataset classes
├── data_module.py         # Data loading and caching
├── model.py               # BERT-CNN model architecture
├── metrics.py             # Metrics computation
├── trainer.py             # Unified trainer
├── training_strategy.py   # Training strategies
├── visualizer.py          # Results visualization
├── main.py                # Main training script
├── inference.py           # Inference script (TODO)
└── requirements.txt       # Python dependencies
```

## Installation

```bash
# Clone repository
git clone <repository-url>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
# Flat classification
python main.py --method flat --dataset large

# Level-wise classification
python main.py --method level --dataset large

# Section-wise classification
python main.py --method section --dataset large
```

### Advanced Options

```bash
python main.py \
  --method flat \
  --dataset large \
  --batch_size 32 \
  --max_epochs 50 \
  --lr 2e-5 \
  --dropout 0.1 \
  --patience 3 \
  --seed 42
```

### Testing Only

```bash
python main.py --method flat --dataset large --test_only
```

## Configuration

Configuration can be managed via:

1. **Command-line arguments** (see `python main.py --help`)
2. **JSON configuration file**:

```json
{
  "model": {
    "bert_model": "indolem/indobert-base-uncased",
    "dropout": 0.1,
    "window_sizes": [1, 2, 3, 4, 5],
    "out_channels": 32
  },
  "training": {
    "method": "flat",
    "batch_size": 32,
    "max_epochs": 50,
    "learning_rate": 2e-5,
    "patience": 3
  },
  "data": {
    "dataset": "large",
    "train_ratio": 0.8,
    "valid_ratio": 0.1,
    "test_ratio": 0.1
  }
}
```

## Performance Tips

1. **Use GPU**: Training will be much faster with CUDA
2. **Adjust batch size**: Increase if you have more GPU memory
3. **Use caching**: Data preprocessing is cached after first run
4. **Early stopping**: Enabled by default to prevent overfitting

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
