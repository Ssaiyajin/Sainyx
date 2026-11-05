---
license: unknown
library_name: pytorch
pipeline_tag: text-to-image
---

# Sainyx image model

## Summary

An experimental PyTorch diffusion image generator. The repository contains both unconditional and tag-conditioned code paths; prompt conditioning depends on the checkpoint loaded at runtime.

## Size

The dataset builder defaults to 64-pixel training images. The checkpoint's parameter count and file size are not tracked, and generated dimensions depend on the loaded checkpoint.

## Intended use

Non-production experimentation and demonstration. Do not use outputs as accurate depictions of real people, events, or products.

## Training data

The checked-in dataset builder retrieves tagged artwork through Safebooru's public API and resizes images for training. The exact dataset snapshot and licensing/redistribution rights for each source image are not documented. The checkpoint's final training data composition is unverified.

## Limitations

The model operates at low native resolution. Image quality is experimental and can include artifacts, anatomy errors, prompt mismatch, and inconsistent style. Upscaling is post-processing and does not restore missing source detail. Prompt interpretation depends on the checkpoint's tag vocabulary.

## Weights and license

Checkpoint redistribution terms and the exact model license are unverified. No license for the repository source code is specified, and it should not be assumed to grant rights to the weights or source artwork.
