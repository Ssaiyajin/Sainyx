"""
generation/video/train.py

Trains VideoUNet on clips instead of independent images. Same resume/
checkpoint/HF-push machinery as before (core.utils.checkpoint_utils),
just pointed at ClipFolderDataset + VideoUNet instead of
ImageFolderDataset + TinyUNet.
"""

import copy
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
    load_checkpoint,
    push_checkpoint_to_hf,
    download_latest_checkpoint_from_hf,
    SessionTimer,
)

# ---- Config -----------------------------------------------------------
IMAGE_SIZE = 64
CLIP_LEN = 16
BATCH_SIZE = 4          # clips per batch; effective tensor is BATCH_SIZE*CLIP_LEN frames
EPOCHS = 100
LR = 2e-4
EMA_DECAY = 0.999
BASE_CH = 64
COND_FRAME = True        # image-to-video: frame 0 is given, the model animates it
TEMPORAL = "v2"          # "mix" = old pooled block, "v2" = per-pixel temporal conv + attention
TIMESTEPS = 1000
SAVE_EVERY_STEPS = 200
DATA_DIR = "/kaggle/working/data/video_clips"   # ClipFolderDataset root (clip_XXXX/ subfolders)
LOCAL_CKPT_DIR = "/kaggle/working/checkpoints"
HF_REPO_ID = "ssaiyajin/sainyx-model"
# Separate folder on purpose: the live app still loads video_gen/sainyx_video_full.pt,
# and an image-to-video model must not resume from (or overwrite) that old model.
HF_CKPT_PATH_IN_REPO = "video_gen_i2v/checkpoint_latest.pt"
HF_FINAL_PATH_IN_REPO = "video_gen_i2v/sainyx_video_i2v.pt"
HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------------

os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)


def save_full(model, ema, optimizer, step, epoch, loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "epoch": epoch, "loss": loss,
        "base_ch": BASE_CH, "temporal": TEMPORAL, "cond_frame": COND_FRAME,
        "image_size": IMAGE_SIZE, "clip_len": CLIP_LEN, "timesteps": TIMESTEPS,
    }, path)


@torch.no_grad()
def update_ema(ema, model):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(EMA_DECAY).add_(pm.detach(), alpha=1 - EMA_DECAY)


def main():
    model = VideoUNet(base_ch=BASE_CH, temporal=TEMPORAL, cond_frame=COND_FRAME).to(DEVICE)
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
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
            ema.load_state_dict(
                torch.load(downloaded_path, map_location=DEVICE).get("ema_state_dict", model.state_dict())
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
            if COND_FRAME:
                # frame 0 is shown to the model clean (and as the cond input),
                # exactly as at sampling time, and only frames 1.. are scored
                first = batch[:, 0]
                noisy_clips = torch.cat([first[:, None], noisy_clips[:, 1:]], dim=1)
                predicted_noise = model(noisy_clips, t, cond=first)
                loss = torch.nn.functional.mse_loss(predicted_noise[:, 1:], noise[:, 1:])
            else:
                predicted_noise = model(noisy_clips, t)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema, model)

            global_step += 1

            if global_step % SAVE_EVERY_STEPS == 0:
                ckpt_path = os.path.join(LOCAL_CKPT_DIR, f"checkpoint_step{global_step}.pt")
                save_full(model, ema, optimizer, global_step, epoch, loss.item(), ckpt_path)
                print(f"[step {global_step}] loss={loss.item():.4f} saved -> {ckpt_path}")

                if HF_TOKEN:
                    push_checkpoint_to_hf(ckpt_path, HF_REPO_ID, HF_CKPT_PATH_IN_REPO, HF_TOKEN)
                    print(f"  pushed to hf://{HF_REPO_ID}/{HF_CKPT_PATH_IN_REPO}")

            if timer.should_stop():
                print(f"Session time limit approaching ({timer.elapsed_minutes():.1f} min elapsed).")
                ckpt_path = os.path.join(LOCAL_CKPT_DIR, "checkpoint_session_end.pt")
                save_full(model, ema, optimizer, global_step, epoch, loss.item(), ckpt_path)
                if HF_TOKEN:
                    push_checkpoint_to_hf(ckpt_path, HF_REPO_ID, HF_CKPT_PATH_IN_REPO, HF_TOKEN)
                    print("Final checkpoint pushed to HF. Safe to let the session end.")
                return

        print(f"Epoch {epoch} complete.")

    final_path = os.path.join(LOCAL_CKPT_DIR, "sainyx_video_full.pt")
    save_full(model, ema, optimizer, global_step, EPOCHS, loss.item(), final_path)
    if HF_TOKEN:
        push_checkpoint_to_hf(final_path, HF_REPO_ID, HF_FINAL_PATH_IN_REPO, HF_TOKEN)
        print(f"Final model pushed to hf://{HF_REPO_ID}/{HF_FINAL_PATH_IN_REPO}")


if __name__ == "__main__":
    main()