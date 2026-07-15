import torch
import torch.nn as nn
import os
import sys
import math
from collections import OrderedDict

# ── Make project root importable (train.py lives in generation/text/) ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from model.gpt import Sainyx, BLOCK_SIZE, VOCAB_SIZE

# ── Device ───────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

# ── Load Data ────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'sainyx_data.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
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
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([d[i:i+BLOCK_SIZE] for i in ix])
    y  = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

# ── Val loss estimation ────────────────────────────
EVAL_ITERS = 100  # batches averaged per val/train loss estimate

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.mean().item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ── LR schedule: linear warmup + cosine decay ──────
MAX_LR       = 1e-3
MIN_LR       = 1e-4
WARMUP_STEPS = 2000

def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

# ── Model ─────────────────────────────────────────
model = Sainyx(vocab_size=vocab_size).to(device)

# Use both GPUs if available
if torch.cuda.device_count() > 1:
    print(f"🔥 Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

# ── Checkpoint paths ──────────────────────────────
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'generation', 'text', 'checkpoints', 'sainyx_checkpoint.pt')
BEST_PATH       = os.path.join(PROJECT_ROOT, 'generation', 'text', 'sainyx_best.pt')
os.makedirs(os.path.join(PROJECT_ROOT, 'generation', 'text', 'checkpoints'), exist_ok=True)

start_step = 0
best_val_loss = float('inf')

# ── Resume from checkpoint if it exists ───────────
if os.path.exists(CHECKPOINT_PATH):
    print(f"\n🔄 Found checkpoint — resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # handle DataParallel prefix mismatch
    if torch.cuda.device_count() > 1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key = k if k.startswith('module.') else 'module.' + k
            new_state_dict[key] = v
        model.load_state_dict(new_state_dict)
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            new_state_dict[key] = v
        model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"✅ Resumed from step {start_step}, last loss: {checkpoint['loss']:.4f}, best val: {best_val_loss:.4f}\n")
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
    lr = get_lr(step, EPOCHS)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    x, y = get_batch('train')
    logits, loss = model(x, y)
    loss = loss.mean()  # DataParallel returns loss per GPU, need to average

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % EVAL_EVERY == 0:
        losses = estimate_loss(model)
        print(f"Step {step:>6} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | LR: {lr:.2e}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'model_state_dict': model.state_dict(),
                'chars': chars,
                'stoi': stoi,
                'itos': itos,
                'step': step,
                'val_loss': best_val_loss,
            }, BEST_PATH)
            print(f"   ✅ New best val loss: {best_val_loss:.4f} — saved sainyx_best.pt")

    # save checkpoint periodically
    if step % SAVE_EVERY == 0 and step > start_step:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'best_val_loss': best_val_loss,
        }, CHECKPOINT_PATH)

print("\nTraining complete!")

# ── Save Final Model ──────────────────────────────
FINAL_PATH = os.path.join(PROJECT_ROOT, 'generation', 'text', 'sainyx_v1.pt')
torch.save(model.state_dict(), FINAL_PATH)
print(f"Model saved to {FINAL_PATH}")

# remove checkpoint since training finished
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint cleared (training finished cleanly)")

# ── Generate Text ─────────────────────────────────
print("\n── Sainyx says: ──────────────────────────")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# access underlying model from DataParallel wrapper
raw_model = model.module if hasattr(model, 'module') else model
generated = raw_model.generate(context, max_new_tokens=300)
print(decode(generated[0].tolist()))