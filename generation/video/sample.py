"""
sample.py
Load a trained checkpoint and generate images (Tier 1) via the reverse
diffusion process. Run this to check "did anything actually learn" -
you're looking for blobby DBZ-color-palette shapes on a v1 checkpoint,
not photorealism.
"""

import os
import sys
import torch
import torchvision.utils as vutils

# Make project root importable when this script is run directly (e.g. on Kaggle)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from model.video_unet import TinyUNet
from generation.video.diffusion import NoiseScheduler
from generation.video.checkpoint_utils import load_checkpoint

CHECKPOINT_PATH = "/kaggle/working/checkpoints/checkpoint_session_end.pt"
IMAGE_SIZE = 64
BATCH_SIZE = 8
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_PATH = "/kaggle/working/samples.png"


def main():
    model = TinyUNet(base_ch=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters())  # needed only to load state dict shape
    load_checkpoint(model, optimizer, CHECKPOINT_PATH, device=DEVICE)
    model.eval()

    scheduler = NoiseScheduler(timesteps=TIMESTEPS, device=DEVICE)

    samples = scheduler.sample(
        model, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, channels=3, device=DEVICE
    )

    # Undo the [-1, 1] normalization from training for viewing
    samples = (samples.clamp(-1, 1) + 1) / 2
    vutils.save_image(samples, OUTPUT_PATH, nrow=4)
    print(f"Saved samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
