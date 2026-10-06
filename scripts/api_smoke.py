"""Run a real HTTP smoke test against the Sainyx text-generation API."""

import os
import secrets
import sys
import threading
from pathlib import Path

import requests
from flask import Flask
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.api import api


def main():
    api_key = secrets.token_urlsafe(24)
    os.environ["SAINYX_API_KEY"] = api_key

    app = Flask("sainyx-api-smoke")
    app.register_blueprint(api)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/api/v1"

    try:
        root_response = requests.get(f"{base_url}", timeout=10)
        root_response.raise_for_status()
        if root_response.json().get("version") != "v1":
            raise RuntimeError("API root did not return the v1 discovery response")

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
            timeout=180,
        )
        response.raise_for_status()

        payload = response.json()
        reply = payload.get("result", {}).get("text", "").strip()
        if payload.get("status") != "completed" or not reply:
            raise RuntimeError(f"Text generation returned no reply: {payload!r}")

        print(f"Sample prompt: {prompt}")
        print(f"Sainyx reply: {reply}")
        print("API smoke test passed: real text generation returned a reply.")
    finally:
        server.shutdown()
        server_thread.join(timeout=5)


if __name__ == "__main__":
    main()