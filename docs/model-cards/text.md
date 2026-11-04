---
license: unknown
library_name: pytorch
pipeline_tag: text-generation
---

# Sainyx text model

## Summary

An experimental, character-level autoregressive transformer implemented in PyTorch. The configured Hub checkpoint filename includes `57m`, but the actual checkpoint is not in this repository and the parameter count has not been verified against the published artifact.

## Size

The configuration describes a seven-layer, 816-dimensional model. The implementation and checkpoint naming do not establish the exact parameter count or checkpoint file size; neither should be advertised as verified.

## Intended use

Experimentation and demonstration of a small text-generation model. Outputs are not suitable for decisions, factual reliance, safety-critical use, or unsupervised publication.

## Training data

The repository's text dataset builder collects Wikipedia extracts and uses local text corpora. The exact corpus snapshot, provenance, filtering, licenses, and final training mixture for the published checkpoint are not established here. Do not describe the data as wholly self-generated or owned.

## Limitations

Answers can be incoherent, incomplete, repetitive, or factually wrong. Character-level generation has limited grounding and instruction following. The model can reproduce patterns from training data.

## Weights and license

Checkpoint redistribution terms and the exact model license are unverified. No license for the repository source code is specified, and it should not be assumed to grant rights to the weights.
