"""Verify Sainyx API readiness after deployment without running model inference."""

import os
import time

import requests


DEFAULT_API_BASE_URL = "https://ssaiyajin-sainyx.hf.space/api/v1"
STARTUP_TIMEOUT_SECONDS = 20 * 60
STARTUP_RETRY_SECONDS = 20


def wait_for_public_api(base_url: str, headers: dict[str, str]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    root_url = base_url.rstrip("/")

    while time.monotonic() < deadline:
        try:
            response = requests.get(root_url, headers=headers, timeout=15)
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
    headers: dict[str, str] = {}
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        headers["Authorization"] = "Bearer " + hf_token

    wait_for_public_api(base_url, headers)
    print("Post-deployment API readiness check passed; no model inference was requested.")


if __name__ == "__main__":
    main()
