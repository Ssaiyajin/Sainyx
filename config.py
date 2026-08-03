"""
config.py
Centralized settings shared across app.py and the generation scripts.
Keeps HF repo IDs and checkpoint locations in one place instead of
duplicated/hardcoded across multiple files.

Checkpoints are intentionally NOT committed to git - they are trained on
Kaggle and pulled from Hugging Face Hub at runtime. If a checkpoint isn't
found locally in CHECKPOINT_DIR, it gets downloaded automatically.
"""

import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.environ.get("HF_TOKEN")

# ── Shared repo IDs ──────────────────────────────────────────────────
# sainyx-model = the "production" repo the live app actually loads from
# sainyx-staging = a backup/pre-promotion copy of everything, so a
# checkpoint always lands in both places the moment training finishes,
# with no manual Kaggle download/upload step in between.
SAINYX_MODEL_REPO_ID = "ssaiyajin/sainyx-model"
SAINYX_STAGING_REPO_ID = "ssaiyajin/sainyx-staging"

# ── Text model (character-level transformer) ──────────────────────────
TEXT_MODEL_REPO_ID = "ssaiyajin/sainyx-model"
TEXT_MODEL_FILENAME = "sainyx_v2_full.pt"
TEXT_MODEL_LOCAL_PATH = os.path.join(CHECKPOINT_DIR, TEXT_MODEL_FILENAME)

# ── Image diffusion model ──────────────────────────────────────────────
IMAGE_MODEL_REPO_ID = "ssaiyajin/sainyx-staging"  # promote to sainyx-model once stable
IMAGE_MODEL_FILENAME = "sainyx_diffusion_full.pt"
IMAGE_MODEL_LOCAL_PATH = os.path.join(CHECKPOINT_DIR, IMAGE_MODEL_FILENAME)
IMAGE_SIZE_DEFAULT = 128
IMAGE_TIMESTEPS_DEFAULT = 1000

# ── Video diffusion model (in development) ─────────────────────────────
VIDEO_MODEL_REPO_ID = "ssaiyajin/sainyx-model"
VIDEO_CHECKPOINT_PATH_IN_REPO = "video_gen/checkpoint_latest.pt"
VIDEO_MODEL_LOCAL_PATH = os.path.join(CHECKPOINT_DIR, "video_checkpoint_latest.pt")
VIDEO_IMAGE_SIZE_DEFAULT = 64
VIDEO_TIMESTEPS_DEFAULT = 1000