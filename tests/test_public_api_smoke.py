from types import SimpleNamespace

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

    public_smoke.wait_for_public_api(
        "https://example.test/api/v1/",
        {"Authorization": "Bearer test"},
    )

    assert requests_made == [
        (
            "https://example.test/api/v1",
            {"headers": {"Authorization": "Bearer test"}, "timeout": 15},
        ),
        (
            "https://example.test/api/v1",
            {"headers": {"Authorization": "Bearer test"}, "timeout": 15},
        ),
    ]


def test_smoke_check_only_checks_readiness_and_authenticates_private_space(monkeypatch):
    monkeypatch.setenv("SAINYX_PUBLIC_API_URL", "https://private.example/api/v1")
    monkeypatch.delenv("SAINYX_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    requests_made = []

    def get(url, **kwargs):
        requests_made.append((url, kwargs))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"version": "v1", "generate_url": "/api/v1/generate"},
        )

    def unexpected_post(*args, **kwargs):
        raise AssertionError("readiness check must not request generation")

    monkeypatch.setattr(public_smoke.requests, "get", get)
    monkeypatch.setattr(public_smoke.requests, "post", unexpected_post)

    public_smoke.main()

    assert requests_made == [
        (
            "https://private.example/api/v1",
            {
                "headers": {"Authorization": "Bearer test-token"},
                "timeout": 15,
            },
        )
    ]


def test_public_api_readiness_works_without_hugging_face_token(monkeypatch):
    monkeypatch.setenv("SAINYX_PUBLIC_API_URL", "https://public.example/api/v1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    requests_made = []

    def get(url, **kwargs):
        requests_made.append((url, kwargs))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"version": "v1", "generate_url": "/api/v1/generate"},
        )

    monkeypatch.setattr(public_smoke.requests, "get", get)

    public_smoke.main()

    assert requests_made == [
        ("https://public.example/api/v1", {"headers": {}, "timeout": 15})
    ]
