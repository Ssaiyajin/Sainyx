"""
Sainyx REST API.

Everything the API needs lives in this one file: API key auth and a
single generalized generation endpoint. Mount it with:

    from api.api import api
    app.register_blueprint(api)

Endpoints (this list never grows - see below):
    GET  /api/v1/status    - which generation types are available right now
    POST /api/v1/generate  - generate something; "type" in the body picks what

Adding a new generation type (world, audio, whatever comes next) means
writing one function and adding one line to GENERATORS. The URL and the
rest of the API surface never change, so client code never has to be
updated just because a new model shipped internally.
"""

import base64
import io
import os
from functools import wraps
from typing import Callable, Dict, NamedTuple, Tuple

import torch
from flask import Blueprint, jsonify, request
from torchvision.utils import save_image

import config
from core.models.factory import get_image_model, get_text_model, get_video_model
from generation.audio.voice import generate_voice_audio

api = Blueprint("api", __name__, url_prefix="/api/v1")


# ── Auth ─────────────────────────────────────────────────────────────────
# Single shared API key from an environment variable - enough for
# "developers hitting my API" at this project's current scale. Swap for
# per-user keys later if it ever needs multiple external consumers with
# separate quotas.

API_KEY_ENV_VAR = "SAINYX_API_KEY"


def require_api_key(view_func: Callable) -> Callable:
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        expected_key = os.environ.get(API_KEY_ENV_VAR)

        if not expected_key:
            # No key configured on the server - fail closed rather than
            # silently letting every request through.
            return jsonify({"error": "API is not configured (missing server API key)"}), 503

        provided_key = request.headers.get("X-API-Key")
        if not provided_key or provided_key != expected_key:
            return jsonify({"error": "Missing or invalid X-API-Key header"}), 401

        return view_func(*args, **kwargs)

    return wrapped


# ── Generators ───────────────────────────────────────────────────────────
# Each generator takes the parsed request body and returns (payload, status_code).
# This is the only place that changes when a new generation type ships.

def _generate_text(body: dict) -> Tuple[dict, int]:
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str):
        return {"error": "prompt must be a string"}, 400
    prompt = prompt.strip()
    if not prompt:
        return {"error": "prompt is required"}, 400

    try:
        max_new_tokens = min(int(body.get("max_tokens", 80)), 500)
    except (TypeError, ValueError):
        return {"error": "max_tokens must be an integer"}, 400

    if max_new_tokens < 1:
        return {"error": "max_tokens must be at least 1"}, 400

    model, vocab = get_text_model()
    encode, decode = vocab["encode"], vocab["decode"]
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        idx = context.clone()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -32:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

    return {"prompt": prompt, "text": decode(idx[0].tolist())}, 200


def _generate_image(body: dict) -> Tuple[dict, int]:
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return {"error": "prompt is required"}, 400

    image_result = get_image_model()
    if image_result is None:
        return {"error": "image model is not available"}, 503

    from generation.image.generate import generate_images

    samples = generate_images(
        image_result["model"],
        image_size=image_result["image_size"],
        timesteps=image_result["timesteps"],
        num_images=1,
        device=config.DEVICE,
    )
    buffer = io.BytesIO()
    save_image(samples, buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"prompt": prompt, "image_base64": image_b64, "source": "sainyx-diffusion"}, 200


def _generate_video(body: dict) -> Tuple[dict, int]:
    video_result = get_video_model()
    if video_result is None:
        return {"error": "video model is not available yet (still training)"}, 503

    from generation.video.diffusion import NoiseScheduler

    scheduler = NoiseScheduler(timesteps=video_result["timesteps"], device=config.DEVICE)
    samples = scheduler.sample(
        video_result["model"],
        image_size=video_result["image_size"],
        batch_size=1,
        channels=3,
        device=config.DEVICE,
    )
    samples = (samples.clamp(-1, 1) + 1) / 2
    buffer = io.BytesIO()
    save_image(samples, buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"image_base64": image_b64, "source": "sainyx-video-tier1"}, 200


def _generate_voice(body: dict) -> Tuple[dict, int]:
    text = body.get("text", body.get("prompt", ""))
    if not isinstance(text, str) or not text.strip():
        return {"error": "text is required"}, 400

    voice = body.get("voice", "neutral")
    if not isinstance(voice, str):
        return {"error": "voice must be a string"}, 400

    audio = generate_voice_audio(text.strip(), voice=voice.strip() or "neutral")
    return {
        "text": text.strip(),
        "voice": voice.strip() or "neutral",
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "mime_type": "audio/wav",
        "source": "sainyx-synthetic-voice",
    }, 200


class GenerationType(NamedTuple):
    generate: Callable[[dict], Tuple[dict, int]]
    is_available: Callable[[], bool]


GENERATORS: Dict[str, GenerationType] = {
    "text": GenerationType(_generate_text, is_available=lambda: True),
    "image": GenerationType(_generate_image, is_available=lambda: get_image_model() is not None),
    "video": GenerationType(_generate_video, is_available=lambda: get_video_model() is not None),
    "voice": GenerationType(_generate_voice, is_available=lambda: True),
}


# ── Routes ───────────────────────────────────────────────────────────────

@api.route("/generate", methods=["POST"])
@require_api_key
def generate():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400
    gen_type = body.get("type", "")
    if not isinstance(gen_type, str):
        return jsonify({"error": "type must be a string"}), 400
    gen_type = gen_type.strip().lower()

    entry = GENERATORS.get(gen_type)
    if entry is None:
        return jsonify({
            "error": f"unknown generation type '{gen_type}'",
            "available_types": list(GENERATORS.keys()),
        }), 400

    if not entry.is_available():
        return jsonify({"error": f"'{gen_type}' is not available right now"}), 503

    payload, status_code = entry.generate(body)
    return jsonify(payload), status_code


@api.route("/status", methods=["GET"])
def status():
    """No auth required - lets a developer check what's live before
    spending a call on a model that isn't ready (e.g. video, mid-training)."""
    return jsonify({
        "available_types": {name: entry.is_available() for name, entry in GENERATORS.items()},
    })