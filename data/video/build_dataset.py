"""
Sainyx Video (Tier 1) — Dataset Builder
Downloads tagged images from Safebooru's public API, resizes them, and
saves them into a flat folder ready for the Tier 1 unconditional video
DDPM (generation/video/train.py + model/video_unet.py).

Tier 1 is just a single-frame diffusion model, so this reuses the same
approach as data/images/build_dataset.py — it's the exact same "curation
instinct" (DBZ/anime stills), just pointed at a separate folder so the
video and image checkpoints don't end up trained on identical data.

IMPORTANT: unlike data/images/build_dataset.py (which nests images under
per-tag subfolders), this saves everything FLAT directly into OUTPUT_DIR,
because ImageFolderDataset in data/video/dataset.py only lists files in
root_dir itself — it does not recurse into subfolders. If you change this
to nest by tag, update ImageFolderDataset to match (see data/images/dataset.py
for the recursive version) or your dataset will silently load 0 images.

Run this in your Kaggle notebook (or locally) — no API key required.
"""

import os
import time
import requests
from PIL import Image
from io import BytesIO

# ── Config ──────────────────────────────────────────
TAGS = ["dragon_ball", "son_goku", "super_saiyan", "vegeta"]   # add/change tags — e.g. "super_saiyan", "vegeta"
IMAGES_PER_TAG = 500                  # how many to try to grab per tag
IMAGE_SIZE = 64                       # matches IMAGE_SIZE in generation/video/train.py
OUTPUT_DIR = "dataset_raw_video"      # kept separate from the image model's dataset_raw
LIMIT_PER_REQUEST = 100               # Safebooru API max per page
SLEEP_BETWEEN_REQUESTS = 1.0          # be polite to the API

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
            break  # no more results

        for post in posts:
            # Safebooru returns file_url directly in most responses
            file_url = post.get("file_url") or post.get("image")
            if file_url:
                if file_url.startswith("//"):
                    file_url = "https:" + file_url
                urls.append(file_url)

        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return urls[:count]


def download_and_process(url, save_path, size):
    """Download one image, convert to RGB, resize, save as PNG."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(save_path, "PNG")
        return True
    except Exception:
        return False


def main():
    total_saved = 0
    seen_urls = set()  # de-dupe across tags so overlapping results aren't saved twice

    for tag in TAGS:
        print(f"\n🔍 Fetching post list for tag: '{tag}'")
        urls = get_post_urls(tag, IMAGES_PER_TAG)
        print(f"  Found {len(urls)} candidate images")

        saved = 0
        for i, url in enumerate(urls):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            save_path = os.path.join(OUTPUT_DIR, f"{tag}_{i:04d}.png")
            if download_and_process(url, save_path, IMAGE_SIZE):
                saved += 1
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(urls)} processed, {saved} saved")
            time.sleep(0.2)  # small delay to avoid hammering image hosts

        print(f"✅ '{tag}': saved {saved} images")
        total_saved += saved

    print(f"\n🔥 Done. Total images saved: {total_saved}")
    print(f"Dataset location: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Point generation/video/train.py's DATA_DIR at this folder "
          f"(or wherever you upload it as a Kaggle dataset).")


if __name__ == "__main__":
    main()