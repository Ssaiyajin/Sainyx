---
license: unknown
library_name: pytorch
pipeline_tag: video-generation
---

# Sainyx video model

## Summary

An experimental temporal diffusion model that produces short, low-resolution clips. Video generation is conditional on a compatible checkpoint being available.

## Size

The configuration defaults to 64-pixel frames, eight frames per clip, and 64 base channels. Checkpoint-specific settings can override these defaults; parameter count and checkpoint file size are not verified.

## Intended use

Research and demonstration only. Do not use for factual, safety-critical, or identity-sensitive applications.

## Training data

The repository includes a synthetic clip builder that applies pan-and-zoom transformations to still images. The code describes optional real-footage training paths, but the provenance and exact composition of any published checkpoint's training data are not verified. Source-image rights are also not established.

## Limitations

The model is not generally text-prompt-conditioned; prompts may be metadata or may contribute an initial frame only for a compatible image-to-video model. Motion, temporal consistency, detail, and clip quality are experimental. A checkpoint may not be available in a deployment.

## Weights and license

Checkpoint redistribution terms and the exact model license are unverified. No license for the repository source code is specified, and it should not be assumed to grant rights to the weights or source footage.
