from __future__ import annotations


def build_image_prompt(prompt: str, asset_type: str = "concept_art", style: str = "anime", aspect_ratio: str = "1:1") -> str:
    asset_label = asset_type.replace("_", " ")
    style_label = style.lower()
    aspect_label = aspect_ratio
    base = prompt.strip()
    if not base:
        base = "high quality fantasy scene"

    return (
        f"{base}, {asset_label}, {style_label} style, clean composition, high detail, "
        f"cinematic lighting, vibrant colors, game-ready design, aspect ratio {aspect_label}"
    )


def build_negative_prompt(asset_type: str = "concept_art", style: str = "anime") -> str:
    return (
        "blurry, low quality, bad anatomy, distorted hands, extra limbs, text, watermark, "
        f"poor composition, oversaturated, noisy, weak silhouette, {style} style clutter"
    )
