# Sainyx model cards

This file summarizes the three experimental model families. Detailed cards are available for the [text model](docs/model-cards/text.md), [image model](docs/model-cards/image.md), and [video model](docs/model-cards/video.md).

| Model | Purpose | Data evidence in this repository | Key limits |
|---|---|---|---|
| Text | Character-level transformer; configured seven layers and 816 dimensions | Builder collects Wikipedia and local text corpora; published checkpoint mixture and rights are unverified | Answers may be incoherent or factually incorrect; parameter/file size unverified |
| Image | Low-resolution diffusion; dataset builder defaults to 64px images | Builder retrieves Safebooru-tagged artwork; source rights and checkpoint mixture are unverified | Experimental quality, low native resolution, possible prompt mismatch; parameter/file size unverified |
| Video | Temporal diffusion; defaults to 64px, eight frames, and 64 base channels | Synthetic pan/zoom clips can be built from image data; published checkpoint provenance is unverified | Experimental motion and detail; prompt conditioning and checkpoint availability vary; parameter/file size unverified |

Model weight licenses, dataset licenses, and parameter counts have not been independently verified. The repository's MIT license applies to repository source code only. Copy the corresponding detailed card into each Hugging Face model repository only after checking that repository's actual checkpoint, metadata, and redistribution terms.
