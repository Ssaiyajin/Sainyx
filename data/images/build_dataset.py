"""
Sainyx Diffusion — Dataset Builder
Downloads tagged images from Safebooru's public API, resizes them,
and saves them into a clean folder ready for diffusion model training.

Run this in your Kaggle notebook (or locally) — no API key required.
"""

import os
import time
import requests
from PIL import Image
from io import BytesIO

# ── Config ──────────────────────────────────────────
TAGS = ["dragon_ball", "son_goku"]   # add/change tags — e.g. "super_saiyan", "vegeta"
IMAGES_PER_TAG = 500                  # how many to try to grab per tag
IMAGE_SIZE = 64                       # start small: 64x64 for the toy DDPM
OUTPUT_DIR = "dataset_raw"
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
    except Exception as e:
        return False


def main():
    total_saved = 0
    for tag in TAGS:
        print(f"\n🔍 Fetching post list for tag: '{tag}'")
        urls = get_post_urls(tag, IMAGES_PER_TAG)
        print(f"  Found {len(urls)} candidate images")

        tag_dir = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(tag_dir, exist_ok=True)

        saved = 0
        for i, url in enumerate(urls):
            save_path = os.path.join(tag_dir, f"{tag}_{i:04d}.png")
            if download_and_process(url, save_path, IMAGE_SIZE):
                saved += 1
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(urls)} processed, {saved} saved")
            time.sleep(0.2)  # small delay to avoid hammering image hosts

        print(f"✅ '{tag}': saved {saved} images to {tag_dir}")
        total_saved += saved

    print(f"\n🔥 Done. Total images saved: {total_saved}")
    print(f"Dataset location: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()