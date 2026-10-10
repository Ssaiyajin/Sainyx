"""
data/video/build_motion_clips.py

Builds image-to-video training clips. Each clip is ONE still image that starts
exactly as the still (frame 0 = the original picture) and then moves:

  * camera: zoom, pan and a small roll, eased in from rest
  * life:   a gentle wave warp (hair/cloth/breathing feel) that grows from zero

No second image is pasted in, so every frame has a single subject. (An earlier
version composited two stills per clip; the model learned that clutter.)
Edges are mirror-padded so there are no black borders.

Output layout is unchanged: clip_XXXXX/frame_00.png ... so ClipFolderDataset and
train.py read it as before. frame_00.png is the conditioning frame.
"""

import math
import os
import random
import numpy as np
from PIL import Image

SRC_DIR = "/kaggle/working/data/images_flat"
OUT_DIR = "/kaggle/working/data/video_clips"
CLIP_LEN = 16
IMAGE_SIZE = 64
CLIPS_PER_STILL = 4
PAD = 0.3          # mirror padding as a fraction of the size
GRID = 4           # wave warp control points per side


def _ease(t):
    return t * t * (3 - 2 * t)          # smoothstep: starts and ends slowly


def make_motion_clip(img, out_dir, clip_len=CLIP_LEN, size=IMAGE_SIZE):
    base = img.convert("RGB").resize((size * 2, size * 2), Image.LANCZOS)   # 2x for quality
    S = base.size[0]
    pad = int(S * PAD)
    arr = np.pad(np.asarray(base), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    padded = Image.fromarray(arr)
    P = padded.size[0]

    zoom_end = random.choice([random.uniform(1.15, 1.45), random.uniform(0.8, 0.92)])
    dx, dy = random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)
    roll_end = random.uniform(-10, 10)
    amp = random.uniform(0.015, 0.04)           # wave size, fraction of image
    freq = random.uniform(1.0, 2.5)             # cycles over the clip
    phase = [(random.uniform(0, 6.28), random.uniform(0, 6.28)) for _ in range(GRID + 1)]

    os.makedirs(out_dir, exist_ok=True)
    for i in range(clip_len):
        t = i / (clip_len - 1)
        e = _ease(t)
        zoom = 1.0 + (zoom_end - 1.0) * e
        roll = math.radians(roll_end * e)
        cx = P / 2 + dx * S * e
        cy = P / 2 + dy * S * e
        half = S / (2 * zoom)

        # source quad (rotated crop window) for each output cell, with a wave
        # displacement on the grid points; wave(t=0) = 0 so frame 0 is untouched
        def src_point(u, v, gi, gj):
            x, y = (u - 0.5) * 2 * half, (v - 0.5) * 2 * half
            xr = x * math.cos(roll) - y * math.sin(roll)
            yr = x * math.sin(roll) + y * math.cos(roll)
            w = amp * S * (math.sin(2 * math.pi * freq * t + phase[gi][0]) - math.sin(phase[gi][0]))
            h = amp * S * (math.sin(2 * math.pi * freq * t + phase[gj][1]) - math.sin(phase[gj][1]))
            return cx + xr + w, cy + yr + h

        mesh = []
        step = S // GRID
        for gj in range(GRID):
            for gi in range(GRID):
                x0, y0 = gi * step, gj * step
                x1, y1 = (S if gi == GRID - 1 else x0 + step), (S if gj == GRID - 1 else y0 + step)
                corners = [(x0, y0, gi, gj), (x0, y1, gi, gj + 1), (x1, y1, gi + 1, gj + 1), (x1, y0, gi + 1, gj)]
                quad = []
                for (ox, oy, a, b) in corners:
                    sx, sy = src_point(ox / S, oy / S, a, b)
                    quad += [sx, sy]
                mesh.append(((x0, y0, x1, y1), quad))
        frame = padded.transform((S, S), Image.MESH, mesh, Image.BILINEAR)
        frame.resize((size, size), Image.LANCZOS).save(os.path.join(out_dir, f"frame_{i:02d}.png"))


def main():
    stills = sorted(
        os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"{len(stills)} source stills in {SRC_DIR}")
    idx = 0
    for path in stills:
        for _ in range(CLIPS_PER_STILL):
            try:
                with Image.open(path) as a:
                    make_motion_clip(a.copy(), os.path.join(OUT_DIR, f"clip_{idx:05d}"))
            except Exception as e:
                print("skip:", e)
                continue
            idx += 1
            if idx % 500 == 0:
                print(f"  {idx} clips")
    print(f"Built {idx} clips -> {OUT_DIR}")


if __name__ == "__main__":
    main()