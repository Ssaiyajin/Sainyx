"""
Data utilities for training and inference
Handles tokenization, batch loading, and data processing
"""

import torch
import os
from pathlib import Path
from typing import Tuple, Optional


class Tokenizer:
    """Character-level tokenizer"""
    
    def __init__(self, chars: list = None, stoi: dict = None, itos: dict = None):
        """
        Initialize tokenizer
        
        Args:
            chars: List of unique characters
            stoi: String to index mapping
            itos: Index to string mapping
        """
        self.chars = chars
        self.stoi = stoi or {}
        self.itos = itos or {}
    
    @classmethod
    def from_text(cls, text: str):
        """Create tokenizer from raw text"""
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(chars, stoi, itos)
    
    def encode(self, s: str) -> list:
        """Encode string to token indices"""
        return [self.stoi.get(c, 0) for c in s]
    
    def decode(self, l: list) -> str:
        """Decode token indices to string"""
        return ''.join([self.itos.get(i, '?') for i in l])
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.stoi)


class DataLoader:
    """Efficient batch data loader"""
    
    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        block_size: int,
        device: str = 'cpu'
    ):
        """
        Initialize data loader
        
        Args:
            data: Encoded token tensor
            batch_size: Batch size
            block_size: Context window size
            device: Device to load batches on
        """
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch"""
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


def load_text_data(
    file_path: Path,
    train_split: float = 0.9,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, Tokenizer]:
    """
    Load and tokenize text data
    
    Args:
        file_path: Path to text file
        train_split: Train/val split ratio
        device: Device for tensors
        
    Returns:
        train_data, val_data, tokenizer
    """
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset size: {len(text):,} characters")
    
    # Create tokenizer
    tokenizer = Tokenizer.from_text(text)
    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")
    
    # Encode
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Train size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")
    
    return train_data, val_data, tokenizer


def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: str
) -> dict:
    """
    Estimate loss on train and validation sets
    
    Args:
        model: Model to evaluate
        train_data: Training data
        val_data: Validation data
        batch_size: Batch size
        block_size: Context window
        eval_iters: Number of iterations to average
        device: Device
        
    Returns:
        Dictionary with train_loss and val_loss
    """
    out = {'train': 0.0, 'val': 0.0}
    model.eval()
    
    with torch.no_grad():
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = []
            for _ in range(eval_iters):
                ix = torch.randint(len(data) - block_size, (batch_size,))
                x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
                y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
                
                _, loss = model(x, y)
                losses.append(loss.item())
            
            out[split] = sum(losses) / len(losses)
    
    model.train()
    return out
