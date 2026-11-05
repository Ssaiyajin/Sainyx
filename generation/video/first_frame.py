"""
generation/video/first_frame.py

Image-to-video needs a starting picture. This draws one with the tag-conditioned
image model, so the video model only has to move something that is already
clear instead of inventing a subject from noise.
"""

import torch
import torch.nn.functional as F

from generation.image.generate import generate_images
from generation.image.tags import resolve_prompt


def first_frame_from_prompt(image_model, image_size, timesteps, prompt, video_size,
                            device, seed=None):
    """Returns (frame [C, H, W] in [-1, 1] at the video model's size, resolved).
    `resolved` is the resolve_prompt dict; when resolved["error"] is set no image
    is drawn and the frame is None."""
    resolved = resolve_prompt(image_model, prompt)
    if resolved["error"]:
        return None, resolved
    img = generate_images(
        image_model, image_size=image_size, timesteps=timesteps, num_images=1,
        device=device, seed=seed, tags=resolved["tags"],
    )[0]                                           # [C, H, W] in [0, 1]
    if img.shape[-1] != video_size:
        img = F.interpolate(img[None], size=(video_size, video_size), mode="area")[0]
    return img.to(device) * 2 - 1, resolved