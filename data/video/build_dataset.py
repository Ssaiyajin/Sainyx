"""
data/video/build_dataset.py

Repurposed: this used to scrape flat single images for a "Tier 1"
single-frame model — that's the exact bug the rest of this pipeline was
fixed to avoid, so it no longer does that.

Now it does the whole thing in one pass: scrape stills from Safebooru
(same source as data/images/build_dataset.py), then immediately apply the
same Ken Burns pan/zoom treatment as build_synthetic_clips.py to turn each
still into a clip_XXXX/frame_00.png...frame_07.png folder — the exact
layout ClipFolderDataset expects.

This exists alongside build_synthetic_clips.py as a second entry point:
use this one when you want fresh stills AND clips in one step (e.g. a new
tag list you haven't scraped before); use build_synthetic_clips.py when
you already have stills on disk (e.g. FLAT_DIR from the image pipeline)
and just want the clip treatment applied to what's already there.

Run this in your Kaggle notebook — no API key required.
"""

import os
import time
import random
import requests
from io import BytesIO
from PIL import Image

# ── Config ──────────────────────────────────────────
TAGS = ["dragon_ball", "son_goku", "super_saiyan", "vegeta"]
IMAGES_PER_TAG = 500
STILL_SIZE = 256              # download/resize stills a bit larger than the
                               # final clip frame size, so the Ken Burns
                               # zoom has real pixels to crop into instead
                               # of upscaling blur
CLIP_IMAGE_SIZE = 64          # matches CLIP_LEN/IMAGE_SIZE elsewhere in the pipeline
CLIP_LEN = 8
ZOOM_RANGE = (1.0, 1.3)
PAN_RANGE = (-0.12, 0.12)
OUTPUT_DIR = "/kaggle/working/data/video_clips"   # same layout/target as build_synthetic_clips.py
LIMIT_PER_REQUEST = 100
SLEEP_BETWEEN_REQUESTS = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_post_urls(tag, count):
    """Query Safebooru API for post metadata, return list of image URLs."""
    urls = []
    page = 0
    while len(urls) < count:
        api_url = (
            "https://safebooru.org/index.php"
            f"?page=dapi&s=post&q=index&json=1"
            f"&tags={tag}&limit={LIMIT_PER_REQUEST}&pid={page}"
        )
        try:
            resp = requests.get(api_url, timeout=15)
            resp.raise_for_status()
            posts = resp.json()
        except Exception as e:
            print(f"  ⚠️ request failed on page {page}: {e}")
            break

        if not posts:
            break

        for post in posts:
            file_url = post.get("file_url") or post.get("image")
            if file_url:
                if file_url.startswith("//"):
                    file_url = "https:" + file_url
                urls.append(file_url)

        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return urls[:count]


def download_still(url, size):
    """Download one image, convert to RGB, resize. Returns a PIL Image or None."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return img
    except Exception:
        return None


def make_clip_from_still(img, clip_out_dir, clip_len=CLIP_LEN, image_size=CLIP_IMAGE_SIZE):
    """Same Ken Burns treatment as build_synthetic_clips.py, applied to an
    already-in-memory PIL Image instead of a path on disk."""
    w, h = img.size
    zoom_end = random.uniform(*ZOOM_RANGE)
    dx = random.uniform(*PAN_RANGE)
    dy = random.uniform(*PAN_RANGE)

    os.makedirs(clip_out_dir, exist_ok=True)

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
        crop.save(os.path.join(clip_out_dir, f"frame_{i:02d}.png"))


def main():
    total_clips = 0
    seen_urls = set()
    clip_idx = 0

    for tag in TAGS:
        print(f"\n🔍 Fetching post list for tag: '{tag}'")
        urls = get_post_urls(tag, IMAGES_PER_TAG)
        print(f"  Found {len(urls)} candidate images")

        saved = 0
        for i, url in enumerate(urls):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            still = download_still(url, STILL_SIZE)
            if still is None:
                continue

            clip_out_dir = os.path.join(OUTPUT_DIR, f"clip_{clip_idx:04d}")
            make_clip_from_still(still, clip_out_dir)
            clip_idx += 1
            saved += 1

            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(urls)} processed, {saved} clips built")
            time.sleep(0.2)

        print(f"✅ '{tag}': built {saved} clips")
        total_clips += saved

    print(f"\n🔥 Done. Total clips built: {total_clips}")
    print(f"Dataset location: {os.path.abspath(OUTPUT_DIR)}")
    print("This is already the layout ClipFolderDataset expects "
          "(clip_XXXX/frame_00.png ... frame_07.png) — point "
          "generation/video/train.py's DATA_DIR straight at it.")


if __name__ == "__main__":
    main()