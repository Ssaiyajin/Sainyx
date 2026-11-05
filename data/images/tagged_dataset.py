"""Loads the tagged dataset written by build_dataset.py into tensors."""

import json
import os

import numpy as np
import torch
from PIL import Image

from generation.image.tags import build_vocab, encode_indices


def load_tagged_dataset(root, image_size, min_tag_count=25, max_tags=512, character_tags=(), vocab=None):
    """
    Returns (images, multihot, vocab):
      images    uint8 tensor [N, 3, S, S]
      multihot  float tensor [N, V] with a 1 for every vocabulary tag on the image
      vocab     list of tag names, ordered by frequency
    By default the vocabulary is built from tags.json. Pass `vocab` to reuse an
    existing one instead (for example the one stored in a checkpoint): tag
    indices must keep their meaning when training resumes, even if the dataset
    was rebuilt in between and its tag counts shifted. Tags outside the given
    vocabulary are ignored.
    """
    with open(os.path.join(root, "tags.json")) as f:
        tags_by_file = json.load(f)

    files = sorted(tags_by_file)
    arrays, kept = [], []
    for rel in files:
        try:
            img = Image.open(os.path.join(root, rel)).convert("RGB")
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), Image.LANCZOS)
            arrays.append(np.asarray(img))
            kept.append(rel)
        except Exception:
            continue

    if vocab is None:
        vocab = build_vocab([tags_by_file[r] for r in kept], character_tags, min_tag_count, max_tags)
    else:
        vocab = list(vocab)
    multihot = torch.zeros(len(kept), len(vocab))
    for row, rel in enumerate(kept):
        idx = encode_indices(tags_by_file[rel], vocab)
        if idx:
            multihot[row, idx] = 1.0

    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    return images, multihot, vocab