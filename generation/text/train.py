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
from core.utils.checkpoint_utils import (
    push_to_both_repos,
    push_checkpoint_to_hf,
    download_latest_checkpoint_from_hf,
    delete_checkpoint_from_hf,
    SessionTimer,
)
import config

# ── Device ───────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

# ── Load Data ────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'text', 'sainyx_data.txt')
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

# ── Checkpoint paths / HF targets ─────────────────
CHECKPOINT_DIR  = os.path.join(PROJECT_ROOT, 'generation', 'text', 'checkpoints')
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'sainyx_checkpoint.pt')
HF_DOWNLOAD_DIR = os.path.join(CHECKPOINT_DIR, 'hf_download')
BEST_PATH       = os.path.join(PROJECT_ROOT, 'generation', 'text', 'sainyx_best.pt')
FINAL_PATH      = os.path.join(PROJECT_ROOT, 'generation', 'text', 'sainyx_v2_full.pt')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Resumable checkpoint (weights + optimizer + step) goes to both repos.
HF_CKPT_TARGETS = [
    (config.SAINYX_MODEL_REPO_ID,   config.TEXT_CHECKPOINT_PATH_IN_REPO),
    (config.SAINYX_STAGING_REPO_ID, config.TEXT_CHECKPOINT_PATH_IN_REPO),
]
# Inference-ready file (weights + vocab), the one ModelFactory loads.
HF_MODEL_TARGETS = [
    (config.SAINYX_MODEL_REPO_ID,   config.TEXT_MODEL_FILENAME),
    (config.SAINYX_STAGING_REPO_ID, config.TEXT_MODEL_FILENAME),
]

raw_model = model.module if hasattr(model, 'module') else model


def inference_state_dict():
    """Weights without the DataParallel 'module.' prefix, so chat.py and
    ModelFactory can load the file no matter how many GPUs trained it."""
    return {k.removeprefix('module.'): v for k, v in raw_model.state_dict().items()}


def atomic_save(obj, path):
    """Write to a temp file first so a killed session can't leave a half-written .pt"""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def find_resume_checkpoint():
    """Newest checkpoint across local disk and both HF repos, by step."""
    found = []
    if os.path.exists(CHECKPOINT_PATH):
        found.append(('local', CHECKPOINT_PATH))
    if config.HF_TOKEN:
        for repo_id, path_in_repo in HF_CKPT_TARGETS:
            try:
                path = download_latest_checkpoint_from_hf(
                    repo_id, path_in_repo, os.path.join(HF_DOWNLOAD_DIR, repo_id.split('/')[-1]),
                    config.HF_TOKEN,
                )
                found.append((f'hf://{repo_id}', path))
            except Exception as e:
                print(f"   no checkpoint on hf://{repo_id}/{path_in_repo} ({type(e).__name__})")
    else:
        print("⚠️  HF_TOKEN not set - can only resume from local disk")

    best = None
    for source, path in found:
        ck = torch.load(path, map_location=device)
        if best is None or ck['step'] > best[1]['step']:
            best = (source, ck)
    return best


start_step = 0
best_val_loss = float('inf')

# ── Resume: newest of local / HF model repo / HF staging repo ──
resume = None if os.environ.get('FRESH_START') == '1' else find_resume_checkpoint()
if resume:
    source, checkpoint = resume
    saved_chars = checkpoint.get('chars')
    if saved_chars is not None and saved_chars != chars:
        raise RuntimeError(
            "Checkpoint vocabulary differs from the current dataset, so the weights "
            "would map to the wrong characters. Rebuild the same dataset, or set "
            "FRESH_START=1 to ignore the checkpoint (it will be overwritten on the next push)."
        )
    raw_model.load_state_dict({k.removeprefix('module.'): v
                               for k, v in checkpoint['model_state_dict'].items()})
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"\n🔄 Resumed from {source} at step {start_step} "
          f"(loss {checkpoint['loss']:.4f}, best val {best_val_loss:.4f})\n")
else:
    print("\n🆕 No checkpoint found - starting fresh\n")

total_params = sum(p.numel() for p in model.parameters())
print(f"Sainyx model loaded")
print(f"Total parameters: {total_params:,}")


def save_and_sync_checkpoint(step, loss_value, push=True):
    """Save the resumable checkpoint locally, then mirror it to HF."""
    atomic_save({
        'step': step,
        'model_state_dict': inference_state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_value,
        'best_val_loss': best_val_loss,
        'chars': chars,
    }, CHECKPOINT_PATH)
    if push and config.HF_TOKEN:
        push_to_both_repos(CHECKPOINT_PATH, targets=HF_CKPT_TARGETS, token=config.HF_TOKEN)


# ── Training Loop ─────────────────────────────────
EPOCHS = 100000
EVAL_EVERY = 5000
SAVE_EVERY = 1000    # local checkpoint every 1000 steps
PUSH_EVERY = 5000    # mirror checkpoint to HF every 5000 steps (multiple of SAVE_EVERY)

timer = SessionTimer(max_session_seconds=12 * 60 * 60, safety_margin_seconds=20 * 60)
finished = True   # flips to False if we stop early for the session limit
last_step, last_loss = start_step, float('nan')

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
    last_step, last_loss = step, loss.item()

    if step % EVAL_EVERY == 0:
        losses = estimate_loss(model)
        print(f"Step {step:>6} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | LR: {lr:.2e}")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            atomic_save({
                'model_state_dict': inference_state_dict(),
                'chars': chars,
                'stoi': stoi,
                'itos': itos,
                'step': step,
                'val_loss': best_val_loss,
            }, BEST_PATH)
            print(f"   ✅ New best val loss: {best_val_loss:.4f} - saved sainyx_best.pt")
            if config.HF_TOKEN:
                push_to_both_repos(BEST_PATH, targets=HF_MODEL_TARGETS, token=config.HF_TOKEN)
            else:
                print("   ⚠️  HF_TOKEN not set - skipping auto-push")

    if step % SAVE_EVERY == 0 and step > start_step:
        save_and_sync_checkpoint(step, last_loss, push=(step % PUSH_EVERY == 0))

    # Leave time to push before Kaggle kills the session
    if timer.should_stop():
        print(f"\n⏰ Session limit close ({timer.elapsed_minutes():.0f} min) - saving and stopping at step {step}")
        finished = False
        break

# ── Wrap up ───────────────────────────────────────
if not finished:
    save_and_sync_checkpoint(last_step, last_loss, push=True)
    print("Checkpoint pushed. Re-run this cell in a new session to continue.")
    sys.exit(0)

print("\nTraining complete!")

# ── Final model: weights + vocab in one file, same format as the best file ──
atomic_save({
    'model_state_dict': inference_state_dict(),
    'chars': chars,
    'stoi': stoi,
    'itos': itos,
    'step': EPOCHS,
    'val_loss': best_val_loss,
}, FINAL_PATH)
print(f"Final model saved to {FINAL_PATH}")

push_ok = False
if config.HF_TOKEN:
    print("📤 Pushing final model to Hugging Face (production + staging)...")
    push_ok = push_to_both_repos(FINAL_PATH, targets=HF_MODEL_TARGETS, token=config.HF_TOKEN)
else:
    print("⚠️  HF_TOKEN not set - download the final model from the Kaggle Output panel instead")

# Training is done: drop the local checkpoint, and the HF resume copies only
# if the final model actually landed in both repos.
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
if push_ok:
    for repo_id, path_in_repo in HF_CKPT_TARGETS:
        delete_checkpoint_from_hf(repo_id, path_in_repo, config.HF_TOKEN)

# ── Generate Text ─────────────────────────────────
print("\n── Sainyx says: ──────────────────────────")
raw_model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = raw_model.generate(context, max_new_tokens=300)
print(decode(generated[0].tolist()))