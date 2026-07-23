"""
train.py
Tier 1 training loop: unconditional tiny DDPM.

Run this on Kaggle. It will:
  1. Try to resume from the latest checkpoint on HF (if one exists)
  2. Train, saving locally every SAVE_EVERY steps
  3. Push each checkpoint to your HF model repo immediately
  4. Watch the session clock and exit gracefully before Kaggle kills it,
     rather than losing whatever progress happened since the last save
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model.unet import TinyUNet
from diffusion import NoiseScheduler
from data.dataset import ImageFolderDataset
from checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    push_checkpoint_to_hf,
    download_latest_checkpoint_from_hf,
    SessionTimer,
)

# ---- Config -----------------------------------------------------------
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
LR = 2e-4
TIMESTEPS = 1000
SAVE_EVERY_STEPS = 200
DATA_DIR = "/kaggle/input/your-dataset-folder"   # point this at your curated images
LOCAL_CKPT_DIR = "/kaggle/working/checkpoints"
HF_REPO_ID = "ssaiyajin/sainyx-model"
HF_CKPT_PATH_IN_REPO = "video_gen/checkpoint_latest.pt"
HF_TOKEN = os.environ.get("HF_TOKEN")            # set this as a Kaggle secret
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------

os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)


def main():
    model = TinyUNet(base_ch=64).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = NoiseScheduler(timesteps=TIMESTEPS, device=DEVICE)

    dataset = ImageFolderDataset(DATA_DIR, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    start_epoch, global_step = 0, 0

    # --- Resume logic: pull latest checkpoint from HF if one exists ---
    local_resume_path = os.path.join(LOCAL_CKPT_DIR, "resumed.pt")
    if HF_TOKEN:
        try:
            downloaded_path = download_latest_checkpoint_from_hf(
                HF_REPO_ID, HF_CKPT_PATH_IN_REPO, LOCAL_CKPT_DIR, HF_TOKEN
            )
            start_epoch, global_step, last_loss = load_checkpoint(
                model, optimizer, downloaded_path, device=DEVICE
            )
            print(f"Resumed from epoch {start_epoch}, step {global_step}, loss {last_loss:.4f}")
        except Exception as e:
            print(f"No checkpoint to resume from (starting fresh): {e}")
    else:
        print("HF_TOKEN not set - training will not resume across sessions.")

    timer = SessionTimer(max_session_seconds=12 * 60 * 60, safety_margin_seconds=20 * 60)

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        for batch in loader:
            batch = batch.to(DEVICE)
            t = torch.randint(0, TIMESTEPS, (batch.shape[0],), device=DEVICE).long()

            noisy_images, noise = scheduler.add_noise(batch, t)
            predicted_noise = model(noisy_images, t)

            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % SAVE_EVERY_STEPS == 0:
                ckpt_path = os.path.join(LOCAL_CKPT_DIR, f"checkpoint_step{global_step}.pt")
                save_checkpoint(model, optimizer, global_step, epoch, loss.item(), ckpt_path)
                print(f"[step {global_step}] loss={loss.item():.4f} saved -> {ckpt_path}")

                if HF_TOKEN:
                    push_checkpoint_to_hf(ckpt_path, HF_REPO_ID, HF_CKPT_PATH_IN_REPO, HF_TOKEN)
                    print(f"  pushed to hf://{HF_REPO_ID}/{HF_CKPT_PATH_IN_REPO}")

            # --- Graceful exit before Kaggle kills the session ---
            if timer.should_stop():
                print(f"Session time limit approaching ({timer.elapsed_minutes():.1f} min elapsed).")
                ckpt_path = os.path.join(LOCAL_CKPT_DIR, "checkpoint_session_end.pt")
                save_checkpoint(model, optimizer, global_step, epoch, loss.item(), ckpt_path)
                if HF_TOKEN:
                    push_checkpoint_to_hf(ckpt_path, HF_REPO_ID, HF_CKPT_PATH_IN_REPO, HF_TOKEN)
                    print("Final checkpoint pushed to HF. Safe to let the session end.")
                return

        print(f"Epoch {epoch} complete.")


if __name__ == "__main__":
    main()