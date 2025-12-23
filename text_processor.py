"""
Text preprocessing utilities.
"""
import re
import string
from typing import List, Optional
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class TextProcessor:
    """Text preprocessing for Indonesian text."""
    
    def __init__(self, use_stopwords: bool = True, min_word_length: int = 2):
        """
        Initialize text processor.
        
        Args:
            use_stopwords: Whether to remove stopwords
            min_word_length: Minimum word length to keep
        """
        self.use_stopwords = use_stopwords
        self.min_word_length = min_word_length
        
        if use_stopwords:
            self.stop_words = set(StopWordRemoverFactory().get_stop_words())
        else:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", text)
        
        # Remove newlines
        text = re.sub(r'\n+', ' ', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove single quotes
        text = re.sub(r"'", '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stopwords and short words
        if self.use_stopwords:
            words = [
                word for word in text.split()
                if word not in self.stop_words and len(word) >= self.min_word_length
            ]
        else:
            words = [
                word for word in text.split()
                if len(word) >= self.min_word_length
            ]
        
        text = ' '.join(words)
        
        return text.strip()
    
    def batch_clean(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]
    
    def get_max_length(self, texts: List[str], extra_length: int = 5) -> int:
        """
        Calculate maximum token length from texts.
        
        Args:
            texts: List of texts
            extra_length: Additional buffer length
            
        Returns:
            Maximum length
        """
        token_lengths = [len(text.split()) for text in texts]
        return max(token_lengths) + extra_length if token_lengths else 512


class TokenizerWrapper:
    """Wrapper for BERT tokenizer with caching."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        """
        Initialize tokenizer wrapper.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> dict:
        """
        Encode text to tokens.
        
        Args:
            text: Input text
            return_tensors: Whether to return tensors ('pt' for PyTorch)
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
    
    def batch_encode(
        self,
        texts: List[str],
        return_tensors: Optional[str] = None
    ) -> dict:
        """
        Encode batch of texts.
        
        Args:
            texts: List of input texts
            return_tensors: Whether to return tensors ('pt' for PyTorch)
            
        Returns:
            Dictionary with batched encodings
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)