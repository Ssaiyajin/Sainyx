"""Verify the deployed public Sainyx API after a Hugging Face deployment."""

import os
import time

import requests


DEFAULT_API_BASE_URL = "https://ssaiyajin-sainyx.hf.space/api/v1"
STARTUP_TIMEOUT_SECONDS = 20 * 60
STARTUP_RETRY_SECONDS = 20


def wait_for_public_api(base_url: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    root_url = base_url.rstrip("/")

    while time.monotonic() < deadline:
        try:
            response = requests.get(root_url, timeout=15)
            if response.status_code == 200:
                discovery = response.json()
                if discovery.get("version") == "v1" and discovery.get("generate_url"):
                    print(f"Public API is ready: {root_url}")
                    return
                raise RuntimeError(f"Unexpected API discovery response: {discovery!r}")

            if response.status_code not in (404, 502, 503, 504):
                raise RuntimeError(
                    f"API readiness check failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            print(f"Public API is restarting (HTTP {response.status_code}); retrying...")
        except requests.RequestException as error:
            print(f"Public API is not reachable yet ({error}); retrying...")

        time.sleep(STARTUP_RETRY_SECONDS)

    raise TimeoutError(f"Public API did not become ready within {STARTUP_TIMEOUT_SECONDS} seconds")


def main() -> None:
    base_url = os.environ.get("SAINYX_PUBLIC_API_URL", DEFAULT_API_BASE_URL).rstrip("/")
    api_key = os.environ.get("SAINYX_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SAINYX_API_KEY GitHub Actions secret is required for the live API check")

    wait_for_public_api(base_url)

    prompt = "In one short sentence, introduce Sainyx."
    response = requests.post(
        f"{base_url}/generate",
        headers={"X-API-Key": api_key},
        json={
            "type": "text",
            "prompt": prompt,
            "max_tokens": 24,
            "seed": 42,
        },
        timeout=(15, 300),
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Public generation endpoint returned HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    payload = response.json()
    reply = payload.get("result", {}).get("text", "").strip()
    if payload.get("status") != "completed" or not reply:
        raise RuntimeError(f"Public text generation returned no reply: {payload!r}")

    print(f"Sample prompt: {prompt}")
    print(f"Sainyx public API reply: {reply}")
    print("Post-deployment public API smoke test passed.")


if __name__ == "__main__":
    main()