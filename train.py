import torch
import torch.nn as nn
from model.gpt import Sainyx, BLOCK_SIZE, VOCAB_SIZE

# ── Device ───────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

# ── Load Data ────────────────────────────────────
with open('data/sainyx_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

# ── Tokenizer (character level) ───────────────────
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary: {vocab_size} unique characters")
print(f"Characters: {''.join(chars)}")

# character to integer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ── Train / Val Split ─────────────────────────────
data = torch.tensor(encode(text), dtype=torch.long)
n    = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Train size: {len(train_data):,} tokens")
print(f"Val size:   {len(val_data):,} tokens")

# ── Batch Loader ──────────────────────────────────
BATCH_SIZE = 16

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x    = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y    = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

# ── Model ─────────────────────────────────────────
model = Sainyx(vocab_size=vocab_size).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nSainyx model loaded")
print(f"Total parameters: {total_params:,}")

# ── Optimizer ─────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ── Training Loop ─────────────────────────────────
EPOCHS = 10000
EVAL_EVERY = 1000

print(f"\nStarting training for {EPOCHS} steps...\n")

for step in range(EPOCHS):
    x, y = get_batch('train')
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % EVAL_EVERY == 0:
        print(f"Step {step:>5} | Loss: {loss.item():.4f}")

print("\nTraining complete!")

# ── Save Model ────────────────────────────────────
torch.save(model.state_dict(), 'sainyx_v1.pt')
print("Model saved to sainyx_v1.pt")

# ── Generate Text ─────────────────────────────────
print("\n── Sainyx says: ──────────────────────────")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=300)
print(decode(generated[0].tolist()))