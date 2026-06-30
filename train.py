import torch
import torch.nn as nn
import os
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
BATCH_SIZE = 64

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x    = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y    = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

# ── Model ─────────────────────────────────────────
model = Sainyx(vocab_size=vocab_size).to(device)

# Use both GPUs if available
if torch.cuda.device_count() > 1:
    print(f"🔥 Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ── Checkpoint paths ──────────────────────────────
CHECKPOINT_PATH = 'checkpoints/sainyx_checkpoint.pt'
os.makedirs('checkpoints', exist_ok=True)

start_step = 0

# ── Resume from checkpoint if it exists ───────────
if os.path.exists(CHECKPOINT_PATH):
    print(f"\n🔄 Found checkpoint — resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    print(f"✅ Resumed from step {start_step}, last loss: {checkpoint['loss']:.4f}\n")
else:
    print("\n🆕 No checkpoint found — starting fresh\n")

total_params = sum(p.numel() for p in model.parameters())
print(f"Sainyx model loaded")
print(f"Total parameters: {total_params:,}")

# ── Training Loop ─────────────────────────────────
EPOCHS = 100000
EVAL_EVERY = 5000
SAVE_EVERY = 1000   # save checkpoint every 1000 steps

print(f"\nStarting training from step {start_step} to {EPOCHS}...\n")

for step in range(start_step, EPOCHS):
    x, y = get_batch('train')
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % EVAL_EVERY == 0:
        print(f"Step {step:>5} | Loss: {loss.item():.4f}")

    # save checkpoint periodically
    if step % SAVE_EVERY == 0 and step > start_step:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, CHECKPOINT_PATH)

print("\nTraining complete!")

# ── Save Final Model ──────────────────────────────
torch.save(model.state_dict(), 'sainyx_v1.pt')
print("Model saved to sainyx_v1.pt")

# remove checkpoint since training finished
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint cleared (training finished cleanly)")

# ── Generate Text ─────────────────────────────────
print("\n── Sainyx says: ──────────────────────────")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=300)
print(decode(generated[0].tolist()))