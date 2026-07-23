"""
Sainyx Text Generation Model
GPT-like transformer for character-level text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model Hyperparameters ──────────────────────────
BLOCK_SIZE = 128        # Context length (max sequence)
N_EMBED = 256          # Embedding dimensions
N_HEADS = 8            # Number of attention heads
N_LAYERS = 6           # Number of transformer blocks
DROPOUT = 0.1          # Dropout rate


class AttentionHead(nn.Module):
    """Single self-attention head"""
    
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel"""
    
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One full transformer block: attention + feed-forward"""
    
    def __init__(self):
        super().__init__()
        head_size = N_EMBED // N_HEADS
        self.self_attention = MultiHeadAttention(N_HEADS, head_size)
        self.feed_forward = FeedForwardNetwork()
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.self_attention(self.ln1(x))
        # Feed-forward with residual connection
        x = x + self.feed_forward(self.ln2(x))
        return x


class Sainyx(nn.Module):
    """
    Sainyx: GPT-like transformer model for text generation
    
    Args:
        vocab_size: Size of the vocabulary
    """
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = BLOCK_SIZE
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        
        # Final layer norm and output head
        self.ln_final = nn.LayerNorm(N_EMBED)
        self.head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T) for training
            
        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Loss if targets provided, else None
        """
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, N_EMBED)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, N_EMBED)
        
        # Combine embeddings
        x = tok_emb + pos_emb  # (B, T, N_EMBED)
        
        # Transformer blocks
        x = self.blocks(x)  # (B, T, N_EMBED)
        
        # Final processing
        x = self.ln_final(x)  # (B, T, N_EMBED)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Generate new tokens
        
        Args:
            idx: Starting indices (B, T)
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Extended indices with generated tokens
        """
        for _ in range(max_new_tokens):
            # Only use the last BLOCK_SIZE tokens (model's context window)
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            # Get logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Get last token's logits
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, next_token), dim=1)
        
        return idx

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
