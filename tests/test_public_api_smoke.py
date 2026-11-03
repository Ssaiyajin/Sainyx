from types import SimpleNamespace

import pytest

import scripts.public_api_smoke as public_smoke


def test_public_api_readiness_retries_until_discovery_is_available(monkeypatch):
    responses = iter([
        SimpleNamespace(status_code=503),
        SimpleNamespace(
            status_code=200,
            json=lambda: {"version": "v1", "generate_url": "/api/v1/generate"},
        ),
    ])
    requests_made = []

    def get(url, **kwargs):
        requests_made.append((url, kwargs))
        return next(responses)

    monkeypatch.setattr(public_smoke.requests, "get", get)
    monkeypatch.setattr(public_smoke.time, "sleep", lambda seconds: None)

    public_smoke.wait_for_public_api("https://example.test/api/v1/", {})

    assert requests_made == [
        ("https://example.test/api/v1", {"headers": {}, "timeout": 15}),
        ("https://example.test/api/v1", {"headers": {}, "timeout": 15}),
    ]


def test_smoke_check_authenticates_to_private_space(monkeypatch):
    monkeypatch.setenv("SAINYX_PUBLIC_API_URL", "https://private.example/api/v1")
    monkeypatch.setenv("SAINYX_API_KEY", "app-key")
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    get_calls = []
    post_calls = []

    def fake_get(url, **kwargs):
        get_calls.append((url, kwargs))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"version": "v1", "generate_url": "/api/v1/generate"},
        )

    def fake_post(url, **kwargs):
        post_calls.append((url, kwargs))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"status": "completed", "result": {"text": "Hello from Sainyx."}},
        )

    monkeypatch.setattr(public_smoke.requests, "get", fake_get)
    monkeypatch.setattr(public_smoke.requests, "post", fake_post)

    public_smoke.main()

    expected_headers = {
        "X-API-Key": "app-key",
        "Authorization": "Bearer hf-token",
    }
    assert get_calls[0][0] == "https://private.example/api/v1"
    assert get_calls[0][1]["headers"] == expected_headers
    assert post_calls[0][0] == "https://private.example/api/v1/generate"
    assert post_calls[0][1]["headers"] == expected_headers


def test_smoke_check_requires_target_space_api_key(monkeypatch):
    monkeypatch.delenv("SAINYX_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SAINYX_API_KEY"):
        public_smoke.main()