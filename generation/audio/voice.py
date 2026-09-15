from __future__ import annotations

import io
import wave
import math


def generate_voice_audio(text: str, voice: str = "neutral") -> bytes:
    """Create a simple WAV buffer for text-to-speech-like output.

    This is intentionally lightweight and dependency-free so the app can expose
    a voice endpoint even in CPU-only environments.
    """
    sample_rate = 22050
    duration = max(0.6, min(3.0, 0.08 * max(1, len(text.split()))))
    frames = int(sample_rate * duration)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for i in range(frames):
            t = i / sample_rate
            base = 0.25 * math.sin(2 * math.pi * 220 * t)
            if voice == "neutral":
                amp = 0.35
            elif voice == "energetic":
                amp = 0.45
            else:
                amp = 0.3
            mod = 0.1 * math.sin(2 * math.pi * 2.2 * t)
            value = int(16000 * amp * (base + mod))
            wav_file.writeframesraw(value.to_bytes(2, byteorder='little', signed=True))

    return buffer.getvalue()
