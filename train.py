import torch
import torch.optim as optim
from pathlib import Path
import sys

from config import (
    DEVICE,
    TRAINING_CONFIG,
    DATA_CONFIG,
    TEXT_MODEL_CONFIG,
    CHECKPOINTS_DIR,
)
from core.models import Sainyx
from core.utils import load_text_data, DataLoader, estimate_loss


def train():
    """Main training loop"""
    
    print(f"Training on: {DEVICE}")
    
    # ── Load Data ────────────────────────────────
    train_data, val_data, tokenizer = load_text_data(
        DATA_CONFIG['training_data_file'],
        train_split=DATA_CONFIG['train_split'],
        device=DEVICE
    )
    
    # ── Create Model ─────────────────────────────
    vocab_size = tokenizer.vocab_size
    model = Sainyx(vocab_size=vocab_size).to(DEVICE)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # ── Multi-GPU Support ────────────────────────
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    # ── Optimizer ────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # ── Training State ───────────────────────────
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / 'sainyx_checkpoint.pt'
    
    start_iter = 0
    best_loss = float('inf')
    
    # ── Resume from checkpoint if exists ─────────
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iter', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed from iteration {start_iter}")
    
    # ── Data Loaders ─────────────────────────────
    train_loader = DataLoader(
        train_data,
        batch_size=TRAINING_CONFIG['batch_size'],
        block_size=TEXT_MODEL_CONFIG['block_size'],
        device=DEVICE
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=TRAINING_CONFIG['batch_size'],
        block_size=TEXT_MODEL_CONFIG['block_size'],
        device=DEVICE
    )
    
    # ── Training Loop ────────────────────────────
    model.train()
    
    for iter_num in range(start_iter, TRAINING_CONFIG['max_iters']):
        
        # ── Periodic Evaluation ──────────────────
        if iter_num % TRAINING_CONFIG['eval_interval'] == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                TRAINING_CONFIG['batch_size'],
                TEXT_MODEL_CONFIG['block_size'],
                TRAINING_CONFIG['eval_iters'],
                DEVICE
            )
            print(f"Iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # ── Save checkpoint if val loss improved ──
            if losses['val'] < best_loss:
                best_loss = losses['val']
                print(f"  Saving checkpoint (val_loss: {best_loss:.4f})")
                
                checkpoint_dict = {
                    'iter': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'chars': tokenizer.chars,
                    'stoi': tokenizer.stoi,
                    'itos': tokenizer.itos,
                }
                torch.save(checkpoint_dict, checkpoint_path)
        
        # ── Training Step ────────────────────────
        x, y = train_loader.get_batch()
        logits, loss = model(x, y)
        
        # ── Backward Pass ────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # ── Gradient Clipping ────────────────────
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            TRAINING_CONFIG['grad_clip']
        )
        
        optimizer.step()
        
        if iter_num % 100 == 0:
            print(f"Iteration {iter_num}: loss = {loss.item():.4f}")
    
    print("Training complete!")
    
    # ── Save final model ─────────────────────────
    final_model_path = CHECKPOINTS_DIR / 'sainyx_final.pt'
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'chars': tokenizer.chars,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos,
    }
    torch.save(final_checkpoint, final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == '__main__':
    train()
