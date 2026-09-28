import time

import pytest
from flask import Flask

import api.api as api_module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(api_module.API_KEY_ENV_VAR, "test-secret")
    app = Flask(__name__)
    app.register_blueprint(api_module.api)
    app.testing = True
    return app.test_client()


def test_generation_requires_configured_and_valid_key(client, monkeypatch):
    monkeypatch.delenv(api_module.API_KEY_ENV_VAR)
    response = client.post("/api/v1/generate", json={"type": "text", "prompt": "hello"})
    assert response.status_code == 503
    assert response.json["error"]["code"] == "api_not_configured"

    monkeypatch.setenv(api_module.API_KEY_ENV_VAR, "test-secret")
    response = client.post("/api/v1/generate", json={"type": "text", "prompt": "hello"})
    assert response.status_code == 401
    assert response.json["error"]["code"] == "unauthorized"


def test_generation_validates_options_before_inference(client):
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "test-secret"},
        json={"type": "image", "prompt": "landscape", "num_images": 5},
    )
    assert response.status_code == 400
    assert response.json["error"]["code"] == "invalid_num_images"


def test_generation_rejects_oversized_body(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_REQUEST_BYTES", 64)
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "test-secret"},
        json={"type": "text", "prompt": "x" * 100},
    )
    assert response.status_code == 413
    assert response.json["error"]["code"] == "request_too_large"


def test_generation_rate_limit_has_retry_after(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_GENERATION_REQUESTS_PER_MINUTE", 1)
    api_module._rate_limit_buckets.clear()
    monkeypatch.setitem(
        api_module.GENERATORS,
        "text",
        api_module.GenerationType(lambda body: ({"text": "ok"}, 200), lambda: True),
    )
    headers = {"X-API-Key": "test-secret"}
    first = client.post(
        "/api/v1/generate", headers=headers, json={"type": "text", "prompt": "hello"}
    )
    second = client.post(
        "/api/v1/generate", headers=headers, json={"type": "text", "prompt": "hello"}
    )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"]
    assert second.json["error"]["code"] == "rate_limit_exceeded"


def test_generation_uses_common_json_envelope(client, monkeypatch):
    payload = {"text": "hello world", "max_tokens": 4, "seed": 12}
    monkeypatch.setitem(
        api_module.GENERATORS,
        "text",
        api_module.GenerationType(lambda body: (payload, 200), lambda: True),
    )
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "test-secret"},
        json={"type": "text", "prompt": "hello", "max_tokens": 4},
    )
    assert response.status_code == 200
    assert response.json == {"type": "text", "status": "completed", "result": payload}


def test_binary_media_response_is_downloadable(client, monkeypatch):
    payload = {
        "prompt": "landscape",
        "_media_bytes": b"png-data",
        "_media_type": "image/png",
        "_filename": "sainyx-image.png",
    }
    monkeypatch.setitem(
        api_module.GENERATORS,
        "image",
        api_module.GenerationType(lambda body: (payload, 200), lambda: True),
    )
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "test-secret"},
        json={"type": "image", "prompt": "landscape", "response_format": "binary"},
    )
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.headers["X-Sainyx-Type"] == "image"
    assert response.headers["Content-Disposition"].endswith('filename=sainyx-image.png')
    assert response.data == b"png-data"


def test_async_job_can_be_polled_and_fetched(client, monkeypatch):
    payload = {"text": "finished"}
    monkeypatch.setitem(
        api_module.GENERATORS,
        "text",
        api_module.GenerationType(lambda body: (payload, 200), lambda: True),
    )
    headers = {"X-API-Key": "test-secret"}
    response = client.post(
        "/api/v1/generate",
        headers=headers,
        json={"type": "text", "prompt": "hello", "async": True},
    )
    assert response.status_code == 202
    job_id = response.json["job_id"]

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        if status.json["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)

    assert status.json["status"] == "completed"
    result = client.get(f"/api/v1/jobs/{job_id}/result", headers=headers)
    assert result.status_code == 200
    assert result.json == {"type": "text", "status": "completed", "result": payload}


def test_status_is_public_and_documents_real_capabilities(client, monkeypatch):
    for name in api_module.GENERATORS:
        monkeypatch.setitem(
            api_module.GENERATORS,
            name,
            api_module.GenerationType(lambda body: ({}, 200), lambda: True),
        )
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    assert response.json["api_version"] == "v1"
    assert response.json["generators"]["image"]["prompt_conditioned"] is False
    assert response.json["generators"]["voice"]["speech_synthesis"] is False


def test_openapi_document_is_served(client):
    response = client.get("/api/v1/openapi.yaml")
    assert response.status_code == 200
    assert response.mimetype == "application/yaml"
    assert b"/jobs/{job_id}/result" in response.data


def test_api_version_root_returns_discovery_links(client):
    response = client.get("/api/v1")
    assert response.status_code == 200
    assert response.json["version"] == "v1"
    assert response.json["status_url"] == "/api/v1/status"
    assert response.json["generate_url"] == "/api/v1/generate"
    assert response.json["openapi_url"] == "/api/v1/openapi.yaml"
