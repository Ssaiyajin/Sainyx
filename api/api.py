"""Versioned Sainyx generation API.

Mount the blueprint with ``app.register_blueprint(api)``. Generation types are
registered in ``GENERATORS``; status, OpenAPI, and authenticated job routes share
the same API version and response conventions.
"""

import base64
import io
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Callable, Dict, NamedTuple, Tuple

from flask import Blueprint, current_app, jsonify, request, send_file, url_for

from generation.audio.voice import generate_voice_audio

api = Blueprint("api", __name__, url_prefix="/api/v1")


# ── Auth ─────────────────────────────────────────────────────────────────
# Single shared API key from an environment variable - enough for
# "developers hitting my API" at this project's current scale. Swap for
# per-user keys later if it ever needs multiple external consumers with
# separate quotas.

API_KEY_ENV_VAR = "SAINYX_API_KEY"
MAX_REQUEST_BYTES = 64 * 1024
MAX_GENERATION_REQUESTS_PER_MINUTE = 60
MAX_PENDING_JOBS = 100
JOB_TTL_SECONDS = 60 * 60

_job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sainyx-api")
_jobs = {}
_jobs_lock = threading.RLock()
_generation_lock = threading.Lock()
_rate_limit_buckets = {}
_rate_limit_lock = threading.Lock()


def get_text_model():
    from core.models.factory import get_text_model as load_text_model

    return load_text_model()


def get_image_model():
    from core.models.factory import get_image_model as load_image_model

    return load_image_model()


def get_video_model():
    from core.models.factory import get_video_model as load_video_model

    return load_video_model()


def _error(code: str, message: str, status_code: int, **details):
    body = {"status": "error", "error": {"code": code, "message": message}}
    if details:
        body["error"].update(details)
    return jsonify(body), status_code


def _check_generation_rate_limit():
    now = time.monotonic()
    client = request.remote_addr or "unknown"
    with _rate_limit_lock:
        expired = [key for key, (started, _) in _rate_limit_buckets.items() if now - started >= 60]
        for key in expired:
            _rate_limit_buckets.pop(key, None)

        started, count = _rate_limit_buckets.get(client, (now, 0))
        if now - started >= 60:
            started, count = now, 0
        if count >= MAX_GENERATION_REQUESTS_PER_MINUTE:
            retry_after = max(1, int(60 - (now - started)))
            response, status_code = _error(
                "rate_limit_exceeded", "Generation request limit exceeded; retry later.", 429
            )
            response.headers["Retry-After"] = str(retry_after)
            return response, status_code
        _rate_limit_buckets[client] = (started, count + 1)
    return None


def require_api_key(view_func: Callable) -> Callable:
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        expected_key = os.environ.get(API_KEY_ENV_VAR)

        if not expected_key:
            # No key configured on the server - fail closed rather than
            # silently letting every request through.
            return _error("api_not_configured", "API is not configured (missing server API key)", 503)

        provided_key = request.headers.get("X-API-Key")
        if not provided_key or provided_key != expected_key:
            return _error("unauthorized", "Missing or invalid X-API-Key header", 401)

        if request.endpoint == "api.generate":
            limited = _check_generation_rate_limit()
            if limited:
                return limited

        return view_func(*args, **kwargs)

    return wrapped


# ── Generators ───────────────────────────────────────────────────────────
# Each generator takes the parsed request body and returns (payload, status_code).
# This is the only place that changes when a new generation type ships.

def _generate_text(body: dict) -> Tuple[dict, int]:
    import torch
    import config

    prompt = body["prompt"].strip()
    max_new_tokens = body.get("max_tokens", 80)
    seed = body.get("seed")

    model, vocab = get_text_model()
    encode, decode = vocab["encode"], vocab["decode"]
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(config.DEVICE)
    generator = torch.Generator(device=config.DEVICE) if seed is not None else None
    if generator is not None:
        generator.manual_seed(seed)

    with torch.no_grad():
        idx = context.clone()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -32:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, next_token), dim=1)

    return {
        "prompt": prompt,
        "text": decode(idx[0].tolist()),
        "max_tokens": max_new_tokens,
        "seed": seed,
    }, 200


def _generate_image(body: dict) -> Tuple[dict, int]:
    import config
    from torchvision.utils import save_image

    prompt = body["prompt"].strip()
    num_images = body.get("num_images", 1)
    seed = body.get("seed")

    image_result = get_image_model()
    if image_result is None:
        return {"error": "image model is not available"}, 503

    from generation.image.generate import generate_images
    from generation.image.tags import resolve_prompt

    resolved = resolve_prompt(image_result["model"], prompt)
    if resolved["error"]:
        return {"error": resolved["error"], "unmatched_words": resolved["unmatched"]}, 422

    samples = generate_images(
        image_result["model"],
        image_size=image_result["image_size"],
        timesteps=image_result["timesteps"],
        num_images=num_images,
        device=config.DEVICE,
        seed=seed,
        tags=resolved["tags"],
    )
    image_bytes = []
    for sample in samples:
        buffer = io.BytesIO()
        save_image(sample.unsqueeze(0), buffer, format="PNG")
        image_bytes.append(buffer.getvalue())

    encoded_images = [base64.b64encode(item).decode("ascii") for item in image_bytes]
    payload = {
        "prompt": prompt,
        "prompt_conditioned": resolved["conditioned"],
        "matched_tags": resolved["tags"],
        "unmatched_words": resolved["unmatched"],
        "count": num_images,
        "seed": seed,
        "source": "sainyx-diffusion",
        "_media_bytes": image_bytes[0],
        "_media_type": "image/png",
        "_filename": "sainyx-image.png",
    }
    if num_images == 1:
        payload["image_base64"] = encoded_images[0]
    else:
        payload["images_base64"] = encoded_images
    return payload, 200


def _generate_video(body: dict) -> Tuple[dict, int]:
    import config
    from generation.video.render import generate_video_bytes

    video_result = get_video_model()
    if video_result is None:
        return {"error": "video model is not available yet (still training)"}, 503

    seed = body.get("seed")
    seconds = max(1.0, min(float(body.get("seconds", 4)), 30.0))
    fmt = "mp4" if body.get("format") == "mp4" else "gif"
    media, fmt, n_frames = generate_video_bytes(
        video_result, config.DEVICE, seconds=seconds, fps=16, seed=seed, steps=30, fmt=fmt
    )
    gif_bytes = media
    return {
        "video_base64": base64.b64encode(media).decode("ascii"),
        "format": fmt,
        "seconds": seconds,
        "frames": n_frames,
        "seed": seed,
        "source": "sainyx-video",
        "_media_bytes": gif_bytes,
        "_media_type": "video/mp4" if fmt == "mp4" else "image/gif",
        "_filename": f"sainyx-video.{fmt}",
    }, 200


def _generate_voice(body: dict) -> Tuple[dict, int]:
    text = body.get("text", body.get("prompt", "")).strip()
    voice = body.get("voice", "neutral").strip().lower() or "neutral"
    audio = generate_voice_audio(text, voice=voice)
    return {
        "text": text,
        "voice": voice,
        "speech_synthesis": False,
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "mime_type": "audio/wav",
        "source": "sainyx-synthetic-voice",
        "_media_bytes": audio,
        "_media_type": "audio/wav",
        "_filename": "sainyx-voice.wav",
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


# ── Validation, media, and job helpers ───────────────────────────────────

def _validated_request(body: dict):
    gen_type = body.get("type")
    if not isinstance(gen_type, str) or not gen_type.strip():
        return None, _error("invalid_type", "type must be one of the supported generation types", 400)
    gen_type = gen_type.strip().lower()
    entry = GENERATORS.get(gen_type)
    if entry is None:
        return None, _error(
            "unknown_type", f"Unknown generation type '{gen_type}'", 400,
            available_types=list(GENERATORS),
        )

    allowed_fields = {
        "text": {"type", "prompt", "max_tokens", "seed", "response_format", "async"},
        "image": {"type", "prompt", "num_images", "seed", "response_format", "async"},
        "video": {"type", "seed", "seconds", "format", "response_format", "async"},
        "voice": {"type", "text", "prompt", "voice", "response_format", "async"},
    }[gen_type]
    unknown = sorted(set(body) - allowed_fields)
    if unknown:
        return None, _error("unknown_fields", "Request contains unsupported fields", 400, fields=unknown)

    response_format = body.get("response_format", "json")
    if response_format not in ("json", "binary"):
        return None, _error("invalid_response_format", "response_format must be 'json' or 'binary'", 400)
    if response_format == "binary" and gen_type == "text":
        return None, _error("unsupported_response_format", "Binary output is only available for media types", 400)

    async_mode = body.get("async", False)
    if not isinstance(async_mode, bool):
        return None, _error("invalid_async", "async must be a boolean", 400)

    prompt_field = "text" if gen_type == "voice" and "text" in body else "prompt"
    if gen_type in ("text", "image", "voice"):
        prompt = body.get(prompt_field, "")
        maximum = 4000 if gen_type == "text" else 2000
        if not isinstance(prompt, str) or not prompt.strip():
            return None, _error("missing_prompt", f"{prompt_field} is required", 400)
        if len(prompt.strip()) > maximum:
            return None, _error("prompt_too_long", f"{prompt_field} must be at most {maximum} characters", 400)

    if gen_type == "text":
        max_tokens = body.get("max_tokens", 80)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 500:
            return None, _error("invalid_max_tokens", "max_tokens must be an integer from 1 to 500", 400)

    if gen_type == "image":
        num_images = body.get("num_images", 1)
        if isinstance(num_images, bool) or not isinstance(num_images, int) or not 1 <= num_images <= 4:
            return None, _error("invalid_num_images", "num_images must be an integer from 1 to 4", 400)
        if response_format == "binary" and num_images != 1:
            return None, _error("binary_batch_not_supported", "Binary output requires num_images=1", 400)

    if "seed" in body:
        seed = body["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
            return None, _error("invalid_seed", "seed must be an integer from 0 to 2^63-1", 400)

    if gen_type == "voice":
        voice = body.get("voice", "neutral")
        if not isinstance(voice, str) or voice.strip().lower() not in ("neutral", "energetic"):
            return None, _error("invalid_voice", "voice must be 'neutral' or 'energetic'", 400)

    return {"type": gen_type, "response_format": response_format, "async": async_mode, "entry": entry}, None


def _deliver_result(gen_type: str, payload: dict, response_format: str):
    if response_format == "binary":
        response = send_file(
            io.BytesIO(payload["_media_bytes"]),
            mimetype=payload["_media_type"],
            as_attachment=True,
            download_name=payload["_filename"],
        )
        response.headers["X-Sainyx-Type"] = gen_type
        response.headers["Cache-Control"] = "no-store"
        return response

    result = {key: value for key, value in payload.items() if not key.startswith("_")}
    return jsonify({"type": gen_type, "status": "completed", "result": result})


def _prune_jobs_locked():
    now = time.time()
    expired = [
        job_id for job_id, job in _jobs.items()
        if job["status"] in ("completed", "failed") and now - job["updated_at"] > JOB_TTL_SECONDS
    ]
    for job_id in expired:
        _jobs.pop(job_id, None)


def _run_job(app, job_id: str, gen_type: str, body: dict, entry: GenerationType):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["updated_at"] = time.time()

    try:
        with _generation_lock:
            payload, status_code = entry.generate(body)
        if status_code == 422:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is not None:
                    job.update(
                        status="failed",
                        error={"code": "prompt_not_understood",
                               "message": payload.get("error", "Prompt not understood.")},
                        updated_at=time.time(),
                    )
            return
        if status_code != 200:
            raise RuntimeError("Generation is unavailable")
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(status="completed", result=payload, updated_at=time.time())
    except Exception:
        app.logger.exception("API generation job failed: %s", job_id)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(
                    status="failed",
                    error={"code": "generation_failed", "message": "Generation failed."},
                    updated_at=time.time(),
                )


def _create_job(gen_type: str, body: dict, response_format: str, entry: GenerationType):
    with _jobs_lock:
        _prune_jobs_locked()
        if len(_jobs) >= MAX_PENDING_JOBS:
            completed = sorted(
                ((job["updated_at"], job_id) for job_id, job in _jobs.items()
                 if job["status"] in ("completed", "failed"))
            )
            while completed and len(_jobs) >= MAX_PENDING_JOBS:
                _, expired_id = completed.pop(0)
                _jobs.pop(expired_id, None)
        if len(_jobs) >= MAX_PENDING_JOBS:
            return _error("job_queue_full", "The generation queue is full; retry later.", 503)

        job_id = uuid.uuid4().hex
        now = time.time()
        _jobs[job_id] = {
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "type": gen_type,
            "response_format": response_format,
        }
        app = current_app._get_current_object()
        _job_executor.submit(_run_job, app, job_id, gen_type, body, entry)

    status_url = url_for("api.job_status", job_id=job_id)
    result_url = url_for("api.job_result", job_id=job_id)
    return jsonify({
        "type": gen_type,
        "status": "queued",
        "job_id": job_id,
        "status_url": status_url,
        "result_url": result_url,
    }), 202


# ── Routes ───────────────────────────────────────────────────────────────

@api.route("", methods=["GET"])
def api_index():
    """Return API discovery links for the version root."""
    return jsonify({
        "name": "Sainyx Generation API",
        "version": "v1",
        "status_url": url_for("api.status"),
        "generate_url": url_for("api.generate"),
        "openapi_url": url_for("api.openapi_document"),
        "job_status_url_template": url_for("api.job_status", job_id="{job_id}"),
        "job_result_url_template": url_for("api.job_result", job_id="{job_id}"),
    })


@api.route("/generate", methods=["POST"])
@require_api_key
def generate():
    if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
        return _error("request_too_large", f"Request body must not exceed {MAX_REQUEST_BYTES} bytes", 413)
    if not request.is_json:
        return _error("invalid_content_type", "Content-Type must be application/json", 415)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _error("invalid_json", "Request body must be a JSON object", 400)

    validated, error_response = _validated_request(body)
    if error_response:
        return error_response
    gen_type = validated["type"]
    entry = validated["entry"]
    if not entry.is_available():
        return _error("generator_unavailable", f"'{gen_type}' is not available right now", 503)

    if validated["async"]:
        return _create_job(gen_type, body, validated["response_format"], entry)

    try:
        with _generation_lock:
            payload, status_code = entry.generate(body)
    except Exception:
        current_app.logger.exception("API generation failed for type %s", gen_type)
        return _error("generation_failed", "Generation failed.", 500)
    if status_code == 422:
        return _error("prompt_not_understood", payload.get("error", "Prompt not understood."), 422,
                      unmatched_words=payload.get("unmatched_words", []))
    if status_code != 200:
        return _error("generation_failed", payload.get("error", "Generation failed."), status_code)
    return _deliver_result(gen_type, payload, validated["response_format"])


@api.route("/jobs/<job_id>", methods=["GET"])
@require_api_key
def job_status(job_id: str):
    with _jobs_lock:
        _prune_jobs_locked()
        job = _jobs.get(job_id)
        if job is None:
            return _error("job_not_found", "Job not found or expired.", 404)
        status = job["status"]
        body = {
            "job_id": job_id,
            "type": job["type"],
            "status": status,
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }
        if status == "completed":
            body["result_url"] = url_for("api.job_result", job_id=job_id)
        if status == "failed":
            body["error"] = job["error"]
    return jsonify(body)


@api.route("/jobs/<job_id>/result", methods=["GET"])
@require_api_key
def job_result(job_id: str):
    with _jobs_lock:
        _prune_jobs_locked()
        job = _jobs.get(job_id)
        if job is None:
            return _error("job_not_found", "Job not found or expired.", 404)
        if job["status"] in ("queued", "running"):
            return _error("job_not_complete", "Job is not complete yet.", 409, status=job["status"])
        if job["status"] == "failed":
            return jsonify({"status": "error", "error": job["error"]}), 500
        gen_type = job["type"]
        payload = job["result"]
        response_format = job["response_format"]
    return _deliver_result(gen_type, payload, response_format)


def _image_model_is_conditioned() -> bool:
    """True only if an image model is already loaded and reads the prompt.
    Looks at the cache and never triggers a model download."""
    try:
        from core.models.factory import ModelFactory
    except ImportError:
        return False

    cached = ModelFactory._cache.get("image")
    return bool(cached and getattr(cached["model"], "tag_vocab", None))


@api.route("/status", methods=["GET"])
def status():
    """Public capability check; does not require a generation API key."""
    available = {name: entry.is_available() for name, entry in GENERATORS.items()}
    return jsonify({
        "api_version": "v1",
        "available_types": available,
        "generators": {
            "text": {"available": available["text"], "max_tokens": 500, "supports_seed": True},
            "image": {
                "available": available["image"], "max_images": 4,
                "supports_seed": True, "prompt_conditioned": _image_model_is_conditioned(),
            },
            "video": {
                "available": available["video"], "supports_seed": True,
                "output": "looping GIF clip",
            },
            "voice": {
                "available": available["voice"], "voices": ["neutral", "energetic"],
                "speech_synthesis": False,
            },
        },
        "limits": {
            "request_bytes": MAX_REQUEST_BYTES,
            "generation_requests_per_minute": MAX_GENERATION_REQUESTS_PER_MINUTE,
            "async_jobs": MAX_PENDING_JOBS,
        },
    })


@api.route("/openapi.yaml", methods=["GET"])
def openapi_document():
    spec_path = os.path.join(os.path.dirname(__file__), "openapi.yaml")
    return send_file(spec_path, mimetype="application/yaml")