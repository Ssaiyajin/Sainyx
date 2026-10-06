"""
generation/video/train.py

Trains VideoUNet on clips instead of independent images. Same resume/
checkpoint/HF-push machinery as before (core.utils.checkpoint_utils),
just pointed at ClipFolderDataset + VideoUNet instead of
ImageFolderDataset + TinyUNet.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from model.video_unet import VideoUNet
from generation.video.diffusion import NoiseScheduler
from data.video.clip_dataset import ClipFolderDataset
from core.utils.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    push_checkpoint_to_hf,
    download_latest_checkpoint_from_hf,
    SessionTimer,
)

# ---- Config -----------------------------------------------------------
IMAGE_SIZE = 64
CLIP_LEN = 8
BATCH_SIZE = 8          # clips per batch; effective tensor is BATCH_SIZE*CLIP_LEN frames
EPOCHS = 100
LR = 2e-4
TIMESTEPS = 1000
SAVE_EVERY_STEPS = 200
DATA_DIR = "/kaggle/working/data/video_clips"   # ClipFolderDataset root (clip_XXXX/ subfolders)
LOCAL_CKPT_DIR = "/kaggle/working/checkpoints"
HF_REPO_ID = "ssaiyajin/sainyx-model"
HF_CKPT_PATH_IN_REPO = "video_gen/checkpoint_latest.pt"
HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------

os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)


def main():
    model = VideoUNet(base_ch=64).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = NoiseScheduler(timesteps=TIMESTEPS, device=DEVICE)

    dataset = ClipFolderDataset(DATA_DIR, image_size=IMAGE_SIZE, clip_len=CLIP_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    start_epoch, global_step = 0, 0

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
            # batch: [B, T, C, H, W]
            batch = batch.to(DEVICE)
            B = batch.shape[0]

            # one timestep PER CLIP (not per frame) so the whole clip is
            # noised/denoised together - this is what makes it a video
            # model rather than T independent image models
            t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()

            noisy_clips, noise = scheduler.add_noise(batch, t)
            predicted_noise = model(noisy_clips, t)

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