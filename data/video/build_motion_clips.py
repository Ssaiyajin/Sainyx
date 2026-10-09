"""
data/video/build_motion_clips.py

Builds training clips that actually move. The old Ken Burns clips were one
still with a slow crop, so every frame was almost the same picture and the
model correctly learned "a clip is one image repeated". Here each clip has two
layers moving independently:

  * background: a still, drifting slowly (parallax)
  * foreground: a second still, scaled down, travelling along a random curved
    path with its own scale change, optional flip, and a little rotation

All frames are generated from stills you already have (SRC_DIR), no external
footage. Output layout is unchanged (clip_XXXX/frame_00.png ...), so
ClipFolderDataset and train.py need no changes beyond CLIP_LEN.
"""

import math
import os
import random
from PIL import Image, ImageOps

SRC_DIR = "/kaggle/working/data/images_flat"
OUT_DIR = "/kaggle/working/data/video_clips"
CLIP_LEN = 16
IMAGE_SIZE = 64
CLIPS_PER_STILL = 3          # different random motion for each still
FG_SCALE = (0.35, 0.6)       # foreground size relative to the frame
SS = 2                       # supersample factor, avoids shimmering edges


def _bg_frame(img, t, zoom_end, dx, dy, size):
    w, h = img.size
    zoom = 1.0 + (zoom_end - 1.0) * t
    cw, ch = w / zoom, h / zoom
    cx, cy = w / 2 + dx * w * t, h / 2 + dy * h * t
    left = min(max(0, cx - cw / 2), w - cw)
    top = min(max(0, cy - ch / 2), h - ch)
    return img.crop((left, top, left + cw, top + ch)).resize((size, size), Image.BILINEAR)


def make_motion_clip(bg_img, fg_img, out_dir, clip_len=CLIP_LEN, size=IMAGE_SIZE):
    S = size * SS
    bg = bg_img.convert("RGB")
    fg = fg_img.convert("RGB")
    if random.random() < 0.5:
        fg = ImageOps.mirror(fg)

    zoom_end = random.uniform(1.0, 1.3)
    bdx, bdy = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)

    s0 = random.uniform(*FG_SCALE)
    s1 = min(0.8, max(0.25, s0 * random.uniform(0.8, 1.3)))
    # path: start/end points plus a sideways bulge so motion is curved
    x0, y0 = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
    x1, y1 = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
    bulge = random.uniform(-0.2, 0.2)
    rot0 = random.uniform(-10, 10)
    rot1 = rot0 + random.uniform(-15, 15)

    os.makedirs(out_dir, exist_ok=True)
    for i in range(clip_len):
        t = i / (clip_len - 1)
        frame = _bg_frame(bg, t, zoom_end, bdx, bdy, S)

        scale = s0 + (s1 - s0) * t
        fw = max(8, int(S * scale))
        sprite = fg.resize((fw, fw), Image.BILINEAR).convert("RGBA")
        sprite = sprite.rotate(rot0 + (rot1 - rot0) * t, resample=Image.BILINEAR, expand=True)

        px = x0 + (x1 - x0) * t - bulge * math.sin(math.pi * t) * (y1 - y0)
        py = y0 + (y1 - y0) * t + bulge * math.sin(math.pi * t) * (x1 - x0)
        left = int(px * S - sprite.width / 2)
        top = int(py * S - sprite.height / 2)
        frame.paste(sprite, (left, top), sprite)
        frame.convert("RGB").resize((size, size), Image.LANCZOS).save(
            os.path.join(out_dir, f"frame_{i:02d}.png")
        )


def main():
    stills = sorted(
        os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"{len(stills)} source stills in {SRC_DIR}")
    idx = 0
    for path in stills:
        for _ in range(CLIPS_PER_STILL):
            other = random.choice(stills)
            try:
                with Image.open(path) as a, Image.open(other) as b:
                    make_motion_clip(a.copy(), b.copy(), os.path.join(OUT_DIR, f"clip_{idx:05d}"))
            except Exception as e:
                print("skip:", e)
                continue
            idx += 1
            if idx % 200 == 0:
                print(f"  {idx} clips")
    print(f"Built {idx} motion clips -> {OUT_DIR}")


if __name__ == "__main__":
    main()