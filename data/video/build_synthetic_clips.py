"""
data/video/build_synthetic_clips.py

Bootstrap dataset for the temporal-mixing block. Real DBZ footage is
copyrighted, so instead of scraping video we manufacture "clips" out of
the static Safebooru stills you already have (data/images/build_dataset.py
output). Each clip is one still, slowly panned/zoomed (Ken Burns effect)
over `clip_len` frames.

This is a stopgap, not a destination: it gives the temporal block genuine
frame-to-frame continuity to learn from *today*, without waiting on a real
video source. Swap SRC_DIR for real extracted footage frames later — the
downstream code (ClipFolderDataset, training) doesn't care where the
frames came from, only that each clip subfolder holds frame_00.png,
frame_01.png, ... in order.

Output layout:
    OUT_DIR/
        clip_0000/frame_00.png ... frame_07.png
        clip_0001/frame_00.png ... frame_07.png
        ...
"""

import os
import random
from PIL import Image

SRC_DIR = "/kaggle/working/data/images_flat"     # folder of static stills
OUT_DIR = "/kaggle/working/data/video_clips"     # synthetic clips written here
CLIP_LEN = 8
IMAGE_SIZE = 64
ZOOM_RANGE = (1.0, 1.3)      # end-of-clip zoom factor, randomized per clip
PAN_RANGE = (-0.12, 0.12)    # fraction of image width/height to drift


def make_clip(src_path, out_dir, clip_len=CLIP_LEN, image_size=IMAGE_SIZE):
    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    zoom_end = random.uniform(*ZOOM_RANGE)
    dx = random.uniform(*PAN_RANGE)
    dy = random.uniform(*PAN_RANGE)

    os.makedirs(out_dir, exist_ok=True)

    for i in range(clip_len):
        t = i / (clip_len - 1)
        zoom = 1.0 + (zoom_end - 1.0) * t

        crop_w, crop_h = w / zoom, h / zoom
        cx = w / 2 + dx * w * t
        cy = h / 2 + dy * h * t

        left = min(max(0, cx - crop_w / 2), w - crop_w)
        top = min(max(0, cy - crop_h / 2), h - crop_h)

        crop = img.crop((left, top, left + crop_w, top + crop_h))
        crop = crop.resize((image_size, image_size), Image.LANCZOS)
        crop.save(os.path.join(out_dir, f"frame_{i:02d}.png"))


def main():
    stills = [
        f for f in os.listdir(SRC_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    print(f"Found {len(stills)} source stills in {SRC_DIR}")

    for idx, fname in enumerate(stills):
        clip_out = os.path.join(OUT_DIR, f"clip_{idx:04d}")
        make_clip(os.path.join(SRC_DIR, fname), clip_out)
        if (idx + 1) % 100 == 0:
            print(f"  ...{idx + 1}/{len(stills)} clips built")

    print(f"Built {len(stills)} synthetic clips -> {OUT_DIR}")


if __name__ == "__main__":
    main()