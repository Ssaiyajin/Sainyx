"""
Sainyx Diffusion — Tagged Dataset Builder

Downloads Dragon Ball art from Safebooru's public API and keeps each post's
tags next to its image. The tags are what the image model is conditioned on,
so a prompt like "gogeta blue" can be matched to images that really show that.

Output layout:
    <out_dir>/<character_tag>/<character_tag>_0000.png ...
    <out_dir>/tags.json      {"<character_tag>/<file>.png": ["tag", ...], ...}

No API key needed. Run from a Kaggle notebook or locally:
    python -m data.images.build_dataset
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
from PIL import Image, ImageOps

from generation.image.tags import EXCLUDE_TAGS

# Character tags to collect. Tags that return nothing are reported and skipped.
CHARACTER_TAGS = [
    "son_goku", "vegeta", "gogeta", "vegito", "son_gohan", "trunks_(dragon_ball)",
    "piccolo", "frieza", "cell_(dragon_ball)", "majin_buu", "krillin", "bulma",
    "broly_(dragon_ball_super)", "son_goten", "beerus", "whis",
    "vegetto", "android_18", "yamcha", "tien_shinhan", "master_roshi", "videl",
    "jiren", "goku_black",
]
IMAGES_PER_CHARACTER = 800
IMAGE_SIZE = 64
LIMIT_PER_REQUEST = 100
SLEEP_BETWEEN_PAGES = 0.5
WORKERS = 8
API = "https://safebooru.org/index.php"


def _post_url(post):
    url = post.get("file_url")
    if not url and post.get("directory") and post.get("image"):
        url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
    if url and url.startswith("//"):
        url = "https:" + url
    return url


def fetch_posts(query_tags, count, session=None):
    """Return up to `count` usable posts as dicts {url, tags} for a Safebooru tag query."""
    session = session or requests.Session()
    posts_out, page = [], 0
    while len(posts_out) < count:
        params = {
            "page": "dapi", "s": "post", "q": "index", "json": 1,
            "tags": query_tags, "limit": LIMIT_PER_REQUEST, "pid": page,
        }
        try:
            resp = session.get(API, params=params, timeout=20)
            resp.raise_for_status()
            posts = resp.json() if resp.text.strip() else []
        except Exception as e:
            print(f"  ⚠️ request failed on page {page}: {e}")
            break
        if not posts:
            break
        for post in posts:
            url = _post_url(post)
            tags = (post.get("tags") or "").split()
            if not url or not tags or EXCLUDE_TAGS.intersection(tags):
                continue
            posts_out.append({"url": url, "tags": tags})
        page += 1
        time.sleep(SLEEP_BETWEEN_PAGES)
    return posts_out[:count]


def download_square(url, size, retries=2):
    """Download one image, crop to a square (biased toward the top, where heads
    usually are), resize, and return an RGB PIL image or None."""
    for _ in range(retries + 1):
        try:
            resp = requests.get(url, timeout=25)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return ImageOps.fit(img, (size, size), Image.LANCZOS, centering=(0.5, 0.3))
        except Exception:
            time.sleep(0.5)
    return None


def build(out_dir="dataset_raw", characters=None, per_character=IMAGES_PER_CHARACTER,
          size=IMAGE_SIZE, workers=WORKERS):
    characters = characters or CHARACTER_TAGS
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()
    tags_by_file, seen_hashes, report = {}, set(), {}

    for character in characters:
        print(f"\n🔍 {character}")
        # Prefer single-character posts: cleaner images and cleaner tags.
        posts = fetch_posts(f"{character} solo", per_character, session)
        if len(posts) < per_character:
            known = {p["url"] for p in posts}
            extra = fetch_posts(character, per_character - len(posts), session)
            posts += [p for p in extra if p["url"] not in known]
        print(f"  {len(posts)} candidate posts")
        if not posts:
            report[character] = 0
            continue

        char_dir = os.path.join(out_dir, character)
        os.makedirs(char_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            images = list(pool.map(lambda p: download_square(p["url"], size), posts))

        saved = 0
        for post, img in zip(posts, images):
            if img is None:
                continue
            digest = hashlib.md5(img.tobytes()).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            name = f"{character}_{saved:04d}.png"
            img.save(os.path.join(char_dir, name), "PNG")
            tags_by_file[f"{character}/{name}"] = post["tags"]
            saved += 1
        report[character] = saved
        print(f"  ✅ saved {saved}")

    with open(os.path.join(out_dir, "tags.json"), "w") as f:
        json.dump(tags_by_file, f)

    print("\n📊 Images per character:")
    for character, n in report.items():
        flag = "" if n >= 200 else "   <- few images; this character will be weak"
        print(f"  {character:32s} {n}{flag}")
    print(f"\n🔥 Done. {len(tags_by_file)} images in {os.path.abspath(out_dir)}")
    return report


if __name__ == "__main__":
    build()