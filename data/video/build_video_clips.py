"""
data/video/build_video_clips.py

Builds training clips from REAL video, so the model sees real motion instead of
a still image being panned. Every clip is a folder:

    vid_00000/frame_00.png ... frame_15.png     (16 frames, SIZE x SIZE)
    vid_00000/clip.wav                          (the matching 2 s of audio, 16 kHz mono)

ClipFolderDataset / train.py read the frames and ignore the wav, so training
works as before. The wav files are kept so an audio model can use the same
clips later.

Sources (both optional, use either or both):
  * URLS: direct links, downloaded into WORK_DIR. These are Blender Foundation
    open movies (CC-BY, attribution required). Links that fail are skipped.
  * VIDEO_DIR: any folder of videos (mp4/mkv/mov/avi/webm), searched
    recursively, e.g. a Kaggle dataset you attached under /kaggle/input.
    Only use footage you have the rights to train on.

Scene cuts are detected and no clip crosses a cut, which matters: a clip that
jumps between two shots teaches the model to flicker.
"""

import json
import os
import subprocess
import urllib.request
import wave

import numpy as np
from PIL import Image

# ------------------------------ config -----------------------------------
URLS = [
    # Verify these still resolve; any that 404 are skipped.
    "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov",
    "https://download.blender.org/durian/movies/Sintel.2010.720p.mkv",
    "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
]
VIDEO_DIR = "/kaggle/input"                  # extra videos you attached, optional
WORK_DIR = "/kaggle/working/video_src"
OUT_DIR = "/kaggle/working/data/video_clips"
CLIP_LEN = 16            # frames per clip, must match train.py
FPS = 8                  # frames per second sampled from the source
SIZE = 64                # output frame size (center-cropped square)
STRIDE = 8               # frames between clip starts (8 = 50% overlap)
CUT_THRESHOLD = 28.0     # mean abs pixel change that counts as a scene cut
MIN_MOTION = 1.5         # skip near-static windows (mean abs change per frame)
MAX_MOTION = 22.0        # skip windows that are mostly flashes/fades
SAVE_AUDIO = True
AUDIO_RATE = 16000
MAX_CLIPS_PER_VIDEO = 6000
# -------------------------------------------------------------------------

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def download(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(url))
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    try:
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"  skipped ({e})")
        return None


def find_videos():
    vids = []
    for url in URLS:
        p = download(url, WORK_DIR)
        if p:
            vids.append(p)
    if VIDEO_DIR and os.path.isdir(VIDEO_DIR):
        for root, _, files in os.walk(VIDEO_DIR):
            vids += [os.path.join(root, f) for f in files if f.lower().endswith(VIDEO_EXT)]
    return sorted(set(vids))


def decode_frames(path):
    """All frames at FPS, center-cropped square, scaled to SIZE. [N, SIZE, SIZE, 3] uint8."""
    vf = f"fps={FPS},crop='min(iw,ih)':'min(iw,ih)',scale={SIZE}:{SIZE}:flags=area"
    cmd = [ffmpeg_exe(), "-v", "error", "-i", path, "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    n = len(raw) // (SIZE * SIZE * 3)
    return np.frombuffer(raw[: n * SIZE * SIZE * 3], np.uint8).reshape(n, SIZE, SIZE, 3)


def decode_audio(path):
    """Mono 16 kHz int16 samples, or None if the file has no audio."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-ac", "1",
           "-ar", str(AUDIO_RATE), "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    except Exception:
        return None
    return np.frombuffer(raw, np.int16) if raw else None


def write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(samples.tobytes())


def segments(frames):
    """Split at scene cuts. Returns list of (start, end) frame index ranges."""
    f = frames.astype(np.int16)
    diffs = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))
    cuts = [0] + [i + 1 for i, d in enumerate(diffs) if d > CUT_THRESHOLD] + [len(frames)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= CLIP_LEN], diffs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    videos = find_videos()
    print(f"{len(videos)} source videos")
    if not videos:
        print("No videos found. Add URLs or attach a video dataset and set VIDEO_DIR.")
        return

    index, clip_id = [], 0
    for vpath in videos:
        try:
            frames = decode_frames(vpath)
        except Exception as e:
            print(f"skip {vpath}: {e}")
            continue
        audio = decode_audio(vpath) if SAVE_AUDIO else None
        segs, diffs = segments(frames)
        made = 0
        for a, b in segs:
            for s in range(a, b - CLIP_LEN + 1, STRIDE):
                motion = float(diffs[s:s + CLIP_LEN - 1].mean())
                if motion < MIN_MOTION or motion > MAX_MOTION:
                    continue
                d = os.path.join(OUT_DIR, f"vid_{clip_id:05d}")
                os.makedirs(d, exist_ok=True)
                for i in range(CLIP_LEN):
                    Image.fromarray(frames[s + i]).save(os.path.join(d, f"frame_{i:02d}.png"))
                t0 = s / FPS
                if audio is not None:
                    lo = int(t0 * AUDIO_RATE)
                    hi = int((t0 + CLIP_LEN / FPS) * AUDIO_RATE)
                    if hi <= len(audio):
                        write_wav(os.path.join(d, "clip.wav"), audio[lo:hi])
                index.append({"clip": f"vid_{clip_id:05d}", "source": os.path.basename(vpath),
                              "start_sec": round(t0, 3), "motion": round(motion, 2)})
                clip_id += 1
                made += 1
                if made >= MAX_CLIPS_PER_VIDEO:
                    break
            if made >= MAX_CLIPS_PER_VIDEO:
                break
        print(f"{os.path.basename(vpath)}: {len(frames)} frames, {len(segs)} shots, {made} clips")

    with open(os.path.join(os.path.dirname(OUT_DIR), "video_clips_index.json"), "w") as f:
        json.dump(index, f)
    print(f"Built {clip_id} clips -> {OUT_DIR}")


if __name__ == "__main__":
    main()