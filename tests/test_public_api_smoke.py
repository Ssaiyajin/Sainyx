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

    def get(url, timeout):
        requests_made.append((url, timeout))
        return next(responses)

    monkeypatch.setattr(public_smoke.requests, "get", get)
    monkeypatch.setattr(public_smoke.time, "sleep", lambda seconds: None)

    public_smoke.wait_for_public_api("https://example.test/api/v1/")

    assert requests_made == [
        ("https://example.test/api/v1", 15),
        ("https://example.test/api/v1", 15),
    ]