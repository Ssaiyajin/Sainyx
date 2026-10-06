"""
generation/video/sample.py

Loads a trained VideoUNet checkpoint and generates an actual clip - T
frames sampled jointly, saved out as a GIF you can watch move. This is
the piece that used to silently degrade to "one PNG": the old sample.py
called TinyUNet on a single-image shape and dumped a grid. This one calls
scheduler.sample(..., num_frames=CLIP_LEN) and writes every frame out in
order.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from model.video_unet import VideoUNet
from generation.video.diffusion import NoiseScheduler
from core.utils.checkpoint_utils import load_checkpoint

CHECKPOINT_PATH = "/kaggle/working/checkpoints/checkpoint_session_end.pt"
IMAGE_SIZE = 64
CLIP_LEN = 8
BATCH_SIZE = 1          # clips to generate
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "/kaggle/working/samples"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = VideoUNet(base_ch=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters())
    load_checkpoint(model, optimizer, CHECKPOINT_PATH, device=DEVICE)
    model.eval()

    scheduler = NoiseScheduler(timesteps=TIMESTEPS, device=DEVICE)

    clips = scheduler.sample(
        model,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        channels=3,
        num_frames=CLIP_LEN,
        device=DEVICE,
    )  # [B, T, C, H, W]

    clips = (clips.clamp(-1, 1) + 1) / 2  # undo training normalization

    try:
        import imageio
        HAVE_IMAGEIO = True
    except ImportError:
        HAVE_IMAGEIO = False
        print("imageio not installed (pip install imageio) - saving PNG frames instead of GIF.")

    for b in range(clips.shape[0]):
        frames_uint8 = [
            (clips[b, t].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            for t in range(clips.shape[1])
        ]

        if HAVE_IMAGEIO:
            gif_path = os.path.join(OUTPUT_DIR, f"clip_{b:02d}.gif")
            imageio.mimsave(gif_path, frames_uint8, duration=0.15)
            print(f"Saved {gif_path} ({len(frames_uint8)} frames)")
        else:
            frame_dir = os.path.join(OUTPUT_DIR, f"clip_{b:02d}")
            os.makedirs(frame_dir, exist_ok=True)
            from PIL import Image
            for t, frame in enumerate(frames_uint8):
                Image.fromarray(frame).save(os.path.join(frame_dir, f"frame_{t:02d}.png"))
            print(f"Saved {len(frames_uint8)} frames -> {frame_dir}")


if __name__ == "__main__":
    main()