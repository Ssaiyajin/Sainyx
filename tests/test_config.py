import importlib
import os

import pytest

pytest.importorskip("torch")

import config


def test_config_reads_hugging_face_token_and_cpu_runtime(monkeypatch):
    original_token = os.environ.get("HF_TOKEN")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    try:
        loaded = importlib.reload(config)
        assert loaded.HF_TOKEN == "test-token"
        assert loaded.DEVICE == "cpu"
        assert loaded.TEXT_MODEL_REPO_ID == "ssaiyajin/sainyx-model"
    finally:
        if original_token is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = original_token
        importlib.reload(config)
