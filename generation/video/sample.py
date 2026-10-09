"""
generation/video/sample.py

Generate a clip or a long video from a trained VideoUNet checkpoint.

    python generation/video/sample.py --seconds 8                  # short GIF
    python generation/video/sample.py --seconds 120 --format mp4   # 2 minutes

Long videos are made by chaining 16-frame chunks: each new chunk is generated
with the last few frames of the previous one pinned (see
NoiseScheduler.sample_ddim), so the video continues instead of restarting.
Runs on a Kaggle T4 in minutes; on CPU expect it to be slow.
"""

import argparse
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from model.video_unet import VideoUNet
from generation.video.render import generate_video_bytes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(path):
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model = VideoUNet(base_ch=ckpt.get("base_ch", 64), temporal=ckpt.get("temporal", "mix")).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "image_size": ckpt.get("image_size", 64),
        "clip_len": ckpt.get("clip_len", 8),
        "timesteps": ckpt.get("timesteps", 1000),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/kaggle/working/checkpoints/sainyx_video_full.pt")
    ap.add_argument("--out", default="/kaggle/working/samples")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps per chunk")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--format", choices=["gif", "mp4"], default="gif")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    result = load(args.checkpoint)
    data, fmt, n = generate_video_bytes(
        result, DEVICE, seconds=args.seconds, fps=args.fps,
        seed=args.seed, steps=args.steps, fmt=args.format,
    )
    path = os.path.join(args.out, f"sainyx_video.{fmt}")
    with open(path, "wb") as f:
        f.write(data)
    print(f"Saved {path}: {n} frames at {args.fps} fps ({n / args.fps:.1f}s)")


if __name__ == "__main__":
    main()