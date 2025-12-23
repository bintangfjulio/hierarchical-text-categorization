"""
Improved BERT-CNN model for hierarchical text classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from typing import List, Optional


class BERTCNN(nn.Module):
    """BERT with CNN layers for text classification."""
    
    def __init__(
        self,
        num_classes: int,
        bert_model: str = 'indolem/indobert-base-uncased',
        dropout: float = 0.1,
        window_sizes: List[int] = [1, 2, 3, 4, 5],
        in_channels: int = 4,
        out_channels: int = 32,
        input_size: int = 768,
        freeze_bert: bool = False,
        use_output_layer: bool = True
    ):
        """
        Initialize BERT-CNN model.
        
        Args:
            num_classes: Number of output classes
            bert_model: Pre-trained BERT model name
            dropout: Dropout rate
            window_sizes: CNN window sizes for different n-grams
            in_channels: Number of BERT hidden states to use
            out_channels: Number of CNN output channels per window
            input_size: BERT hidden size
            freeze_bert: Whether to freeze BERT parameters
            use_output_layer: Whether to include final linear layer
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.window_sizes = window_sizes
        self.out_channels = out_channels
        self.use_output_layer = use_output_layer
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(
            bert_model,
            output_hidden_states=True
        )
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Convolutional layers for different n-grams
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(window_size, input_size)
            )
            for window_size in window_sizes
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        feature_dim = len(window_sizes) * out_channels
        self.feature_dim = feature_dim
        
        if use_output_layer:
            self.output_layer = nn.Linear(feature_dim, num_classes)
        else:
            self.output_layer = None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, num_classes] or features [batch_size, feature_dim]
        """
        # BERT encoding
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get last 4 hidden states
        hidden_states = bert_output.hidden_states
        hidden_states = torch.stack(hidden_states[-4:], dim=1)
        # Shape: [batch_size, 4, seq_len, hidden_size]
        
        # Apply convolution + pooling for each window size
        pooled_outputs = []
        for conv_layer in self.conv_layers:
            # Convolution
            conv_out = F.relu(conv_layer(hidden_states).squeeze(3))
            # Shape: [batch_size, out_channels, seq_len - window_size + 1]
            
            # Max pooling
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            # Shape: [batch_size, out_channels]
            
            pooled_outputs.append(pooled)
        
        # Concatenate all pooled features
        features = torch.cat(pooled_outputs, dim=1)
        # Shape: [batch_size, len(window_sizes) * out_channels]
        
        # Apply dropout
        features = self.dropout(features)
        
        # Output layer
        if self.use_output_layer:
            logits = self.output_layer(features)
            return logits
        else:
            return features
    
    def get_feature_dim(self) -> int:
        """Get feature dimension for transfer learning."""
        return self.feature_dim


class HierarchicalBERTCNN(nn.Module):
    """BERT-CNN with hierarchical output heads."""
    
    def __init__(
        self,
        num_classes_per_level: List[int],
        bert_model: str = 'indolem/indobert-base-uncased',
        dropout: float = 0.1,
        window_sizes: List[int] = [1, 2, 3, 4, 5],
        out_channels: int = 32,
        shared_encoder: bool = True
    ):
        """
        Initialize hierarchical model.
        
        Args:
            num_classes_per_level: List of number of classes at each level
            bert_model: Pre-trained BERT model name
            dropout: Dropout rate
            window_sizes: CNN window sizes
            out_channels: CNN output channels
            shared_encoder: Whether to share encoder across levels
        """
        super().__init__()
        
        self.num_levels = len(num_classes_per_level)
        self.shared_encoder = shared_encoder
        
        if shared_encoder:
            # Single shared encoder
            self.encoder = BERTCNN(
                num_classes=0,  # Dummy value
                bert_model=bert_model,
                dropout=dropout,
                window_sizes=window_sizes,
                out_channels=out_channels,
                use_output_layer=False
            )
            
            # Separate output head for each level
            feature_dim = self.encoder.get_feature_dim()
            self.output_heads = nn.ModuleList([
                nn.Linear(feature_dim, num_classes)
                for num_classes in num_classes_per_level
            ])
        else:
            # Separate encoder for each level
            self.encoders = nn.ModuleList([
                BERTCNN(
                    num_classes=num_classes,
                    bert_model=bert_model,
                    dropout=dropout,
                    window_sizes=window_sizes,
                    out_channels=out_channels,
                    use_output_layer=True
                )
                for num_classes in num_classes_per_level
            ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        level: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            level: Specific level to predict (None for all levels)
            attention_mask: Attention mask
            
        Returns:
            Logits for specified level or all levels
        """
        if self.shared_encoder:
            # Get shared features
            features = self.encoder(input_ids, attention_mask)
            
            if level is not None:
                # Single level output
                return self.output_heads[level](features)
            else:
                # All levels output
                outputs = [head(features) for head in self.output_heads]
                return outputs
        else:
            if level is not None:
                # Single level output
                return self.encoders[level](input_ids, attention_mask)
            else:
                # All levels output
                outputs = [
                    encoder(input_ids, attention_mask)
                    for encoder in self.encoders
                ]
                return outputs


def create_model(
    num_classes: int,
    config,
    use_output_layer: bool = True
) -> BERTCNN:
    """
    Factory function to create model from config.
    
    Args:
        num_classes: Number of output classes
        config: Configuration object
        use_output_layer: Whether to include output layer
        
    Returns:
        Initialized model
    """
    return BERTCNN(
        num_classes=num_classes,
        bert_model=config.model.bert_model,
        dropout=config.model.dropout,
        window_sizes=config.model.window_sizes,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        input_size=config.model.input_size,
        use_output_layer=use_output_layer
    )