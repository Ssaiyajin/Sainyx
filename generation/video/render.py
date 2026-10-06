"""
generation/video/render.py

Sampling and GIF encoding for the app and API, kept in one place so
app.py and api/api.py don't each carry their own copy. Uses Pillow only.
"""

import io

import torch
from PIL import Image

from generation.video.diffusion import NoiseScheduler


def clip_to_gif_bytes(clip, scale=3, ms_per_frame=150):
    """clip: [T, C, H, W] tensor in [-1, 1]. Returns GIF bytes that loop forever."""
    arr = ((clip.clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    size = (arr.shape[2] * scale, arr.shape[1] * scale)
    frames = [Image.fromarray(a).resize(size, Image.NEAREST) for a in arr]
    buffer = io.BytesIO()
    frames[0].save(
        buffer, format="GIF", save_all=True, append_images=frames[1:],
        duration=ms_per_frame, loop=0,
    )
    return buffer.getvalue()


def generate_clip_gif(video_result, device, seed=None):
    """Sample one clip from a loaded video model (see ModelFactory.load_video_model)
    and return it as GIF bytes."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    scheduler = NoiseScheduler(timesteps=video_result["timesteps"], device=device)
    clips = scheduler.sample(
        video_result["model"],
        image_size=video_result["image_size"],
        batch_size=1,
        channels=3,
        num_frames=video_result["clip_len"],
        device=device,
        generator=generator,
    )
    return clip_to_gif_bytes(clips[0])