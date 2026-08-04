"""
data/video/clip_dataset.py

The dataset piece that was actually missing. ImageFolderDataset (existing)
returns one image per sample -> no temporal signal, ever, no matter what
the model looks like. ClipFolderDataset returns a whole sequence of frames
per sample, shape [T, C, H, W], which is the minimum requirement for a
temporal model to have anything to learn from.

Expects:
    root_dir/
        clip_0000/frame_00.png, frame_01.png, ...
        clip_0001/frame_00.png, frame_01.png, ...
        ...
(exactly what build_synthetic_clips.py produces, and what real extracted
video frames should also be laid out as)
"""

import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ClipFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=64, clip_len=8):
        self.clip_len = clip_len
        self.clips = []

        for clip_name in sorted(os.listdir(root_dir)):
            clip_path = os.path.join(root_dir, clip_name)
            if not os.path.isdir(clip_path):
                continue
            frames = sorted(
                f for f in os.listdir(clip_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            if len(frames) >= clip_len:
                self.clips.append([os.path.join(clip_path, f) for f in frames])

        print(f"Found {len(self.clips)} clips (>= {clip_len} frames) in {root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frame_paths = self.clips[idx]
        # random contiguous window in case a clip is longer than clip_len
        max_start = len(frame_paths) - self.clip_len
        start = random.randint(0, max_start) if max_start > 0 else 0
        window = frame_paths[start:start + self.clip_len]

        frames = [self.transform(Image.open(p).convert("RGB")) for p in window]
        return torch.stack(frames, dim=0)  # [T, C, H, W]