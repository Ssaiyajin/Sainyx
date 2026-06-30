import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters ──────────────────────────────
VOCAB_SIZE  = 677    # unique characters
BLOCK_SIZE  = 128    # context length
N_EMBED     = 256   # embedding dimensions
N_HEADS     = 8     # attention heads
N_LAYERS    = 6     # transformer blocks
DROPOUT     = 0.1
# ─────────────────────────────────────────────────


class Head(nn.Module):
    """Single self-attention head"""
    def __init__(self, head_size):
        super().__init__()
        self.key    = nn.Linear(N_EMBED, head_size, bias=False)
        self.query  = nn.Linear(N_EMBED, head_size, bias=False)
        self.value  = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Simple neural network after attention"""
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


class Block(nn.Module):
    """One full transformer block: attention + feedforward"""
    def __init__(self):
        super().__init__()
        head_size = N_EMBED // N_HEADS
        self.sa   = MultiHeadAttention(N_HEADS, head_size)
        self.ff   = FeedForward()
        self.ln1  = nn.LayerNorm(N_EMBED)
        self.ln2  = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # attention
        x = x + self.ff(self.ln2(x))   # feedforward
        return x


class Sainyx(nn.Module):
    """The full model"""
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks             = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln_final           = nn.LayerNorm(N_EMBED)
        self.head               = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs  = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx