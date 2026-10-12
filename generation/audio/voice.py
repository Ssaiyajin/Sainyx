from __future__ import annotations

import io
import logging
import math
import os
import threading
import wave
from collections import OrderedDict

log = logging.getLogger(__name__)

# Kokoro-82M (Apache 2.0) renders speech at 24 kHz. The API's two voice names map
# to a British male voice with a calm delivery and a slightly faster one.
KOKORO_SAMPLE_RATE = 24000
VOICE_PRESETS = {
    "neutral": ("bm_george", 1.0),
    "energetic": ("bm_lewis", 1.12),
}
MAX_CHARS = int(os.environ.get("SAINYX_VOICE_MAX_CHARS", "600"))

_pipelines: dict = {}
_lock = threading.Lock()
_kokoro_missing = False

# Small in-memory cache so a repeated phrase (greetings, the same answer twice) is instant.
_CACHE_SIZE = int(os.environ.get("SAINYX_VOICE_CACHE", "32"))
_cache: "OrderedDict[tuple, bytes]" = OrderedDict()


def _get_pipeline(lang_code: str):
    if lang_code not in _pipelines:
        from kokoro import KPipeline  # imported lazily so the app still starts without it

        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


def _to_wav(samples, sample_rate: int) -> bytes:
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _synthesize_kokoro(text: str, voice: str) -> bytes:
    import numpy as np

    kokoro_voice, speed = VOICE_PRESETS.get(voice, VOICE_PRESETS["neutral"])
    chunks = []
    with _lock:  # one synthesis at a time; the pipeline is not guaranteed thread-safe
        pipeline = _get_pipeline(kokoro_voice[0])
        for _, _, audio in pipeline(text, voice=kokoro_voice, speed=speed):
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        raise RuntimeError("Kokoro returned no audio")
    return _to_wav(np.concatenate(chunks), KOKORO_SAMPLE_RATE)


def _tone(text: str, voice: str) -> bytes:
    """Dependency-free placeholder: a short tone. Used only when Kokoro is unavailable."""
    sample_rate = 22050
    duration = max(0.6, min(3.0, 0.08 * max(1, len(text.split()))))
    frames = int(sample_rate * duration)
    amp = {"neutral": 0.35, "energetic": 0.45}.get(voice, 0.3)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(frames):
            t = i / sample_rate
            base = 0.25 * math.sin(2 * math.pi * 220 * t)
            mod = 0.1 * math.sin(2 * math.pi * 2.2 * t)
            value = int(16000 * amp * (base + mod))
            wav_file.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return buffer.getvalue()


def generate_voice(text: str, voice: str = "neutral") -> tuple[bytes, str]:
    """Return (wav_bytes, backend) where backend is "kokoro" or "tone"."""
    global _kokoro_missing
    wanted = os.environ.get("SAINYX_VOICE_BACKEND", "kokoro").lower()
    if wanted != "tone" and not _kokoro_missing and text.strip():
        key = (text[:MAX_CHARS], voice)
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key], "kokoro"
        try:
            audio = _synthesize_kokoro(key[0], voice)
            _cache[key] = audio
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
            return audio, "kokoro"
        except ImportError:
            _kokoro_missing = True
            log.warning("kokoro is not installed, falling back to the placeholder tone")
        except Exception:
            log.exception("Kokoro synthesis failed, falling back to the placeholder tone")
    return _tone(text, voice), "tone"


def generate_voice_audio(text: str, voice: str = "neutral") -> bytes:
    return generate_voice(text, voice)[0]


def warm_up() -> None:
    """Load Kokoro in a background thread so the first real request is not the slow one."""
    if os.environ.get("SAINYX_VOICE_BACKEND", "kokoro").lower() == "tone":
        return

    def _run():
        try:
            _synthesize_kokoro("Ready.", "neutral")
            log.info("Kokoro voice warmed up")
        except Exception:
            log.warning("Kokoro warm-up skipped", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()