from generation.image.prompt_builder import build_image_prompt, build_negative_prompt
from generation.audio.voice import generate_voice_audio


def test_build_image_prompt_includes_asset_type_and_style():
    prompt = build_image_prompt(
        "hero character",
        asset_type="game_asset",
        style="anime",
        aspect_ratio="1:1",
    )
    lowered = prompt.lower()
    assert "game asset" in lowered
    assert "anime" in lowered
    assert "hero character" in lowered


def test_build_negative_prompt_contains_common_artifacts():
    prompt = build_negative_prompt(asset_type="concept_art", style="anime")
    lowered = prompt.lower()
    assert "blurry" in lowered
    assert "low quality" in lowered


def test_generate_voice_audio_creates_wav_bytes():
    audio_bytes = generate_voice_audio("hello world", voice="neutral")
    assert audio_bytes.startswith(b"RIFF")
