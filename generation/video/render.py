"""
generation/video/render.py

Sampling and GIF encoding for the app and API, kept in one place so
app.py and api/api.py don't each carry their own copy. Uses Pillow only.
"""

import io
import os
import tempfile

import numpy as np
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


def _interpolate(frames, factor):
    """frames: [N, C, H, W]. Cheap temporal upsampling by linear blending, so a
    model that outputs 8 fps can be played back at 16 or 24 fps without looking
    like a slideshow."""
    if factor <= 1:
        return frames
    out = []
    for a, b in zip(frames[:-1], frames[1:]):
        for k in range(factor):
            w = k / factor
            out.append((1 - w) * a + w * b)
    out.append(frames[-1])
    return torch.stack(out)


@torch.no_grad()
def generate_long_frames(video_result, device, seconds=4.0, model_fps=8, overlap=4,
                         steps=40, seed=None, first_frame=None):
    """Chain clips into one long frame sequence [N, C, H, W] in [-1, 1].

    The first chunk is unconditional; every later chunk is generated with the
    last `overlap` frames of the previous chunk pinned (see sample_ddim), then
    only its new frames are appended."""
    model = video_result["model"]
    size = video_result["image_size"]
    chunk = video_result["clip_len"]
    overlap = min(overlap, chunk - 1)
    total = max(chunk, int(round(seconds * model_fps)))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    scheduler = NoiseScheduler(timesteps=video_result["timesteps"], device=device)
    i2v = getattr(model, "cond_frame", False)
    if i2v and first_frame is None:
        raise ValueError("this video model animates a first frame; pass first_frame")

    if first_frame is not None and i2v:
        ff = first_frame.to(device)[None]                  # [1, C, H, W]
        frames = scheduler.sample_ddim(
            model, size, chunk, steps=steps, known=ff[:, None], cond=ff,
            device=device, generator=generator,
        )[0]
    else:
        frames = scheduler.sample_ddim(
            model, size, chunk, steps=steps, device=device, generator=generator
        )[0]                                               # [chunk, C, H, W]

    while frames.shape[0] < total:
        known = frames[-overlap:][None]
        nxt = scheduler.sample_ddim(
            model, size, chunk, steps=steps, known=known,
            cond=known[:, 0] if i2v else None, device=device, generator=generator
        )[0]
        frames = torch.cat([frames, nxt[overlap:]], dim=0)
    return frames[:total]


def frames_to_gif_bytes(frames, fps=16, scale=3):
    arr = ((frames.clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    size = (arr.shape[2] * scale, arr.shape[1] * scale)
    imgs = [Image.fromarray(a).resize(size, Image.NEAREST) for a in arr]
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    return buf.getvalue()


def frames_to_mp4_bytes(frames, fps=16, scale=4):
    """H.264 MP4 via imageio-ffmpeg. Raises ImportError if it is not installed,
    so callers can fall back to GIF."""
    import imageio.v2 as imageio
    arr = ((frames.clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    size = (arr.shape[2] * scale, arr.shape[1] * scale)
    imgs = [np.asarray(Image.fromarray(a).resize(size, Image.NEAREST)) for a in arr]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.mp4")
        imageio.mimsave(path, imgs, fps=fps, codec="libx264", quality=8,
                        macro_block_size=1, pixelformat="yuv420p")
        with open(path, "rb") as f:
            return f.read()


def generate_video_bytes(video_result, device, seconds=4.0, fps=16, seed=None,
                         steps=40, fmt="gif", first_frame=None):
    """Main entry for app/API/CLI. Returns (bytes, format_used, n_frames)."""
    model_fps = 8
    frames = generate_long_frames(video_result, device, seconds=seconds,
                                  model_fps=model_fps, steps=steps, seed=seed,
                                  first_frame=first_frame)
    frames = _interpolate(frames, max(1, fps // model_fps))
    if fmt == "mp4":
        try:
            return frames_to_mp4_bytes(frames, fps=fps), "mp4", frames.shape[0]
        except Exception:
            pass
    return frames_to_gif_bytes(frames, fps=fps), "gif", frames.shape[0]