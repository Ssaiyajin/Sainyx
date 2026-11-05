import pytest

pytest.importorskip("torch")

from core.models.factory import ModelFactory


@pytest.fixture(autouse=True)
def clear_model_factory_cache():
    ModelFactory.clear_cache()
    yield
    ModelFactory.clear_cache()


def test_ensure_local_downloads_and_caches_checkpoint_path(tmp_path, monkeypatch):
    source = tmp_path / "hub-checkpoint.pt"
    source.write_bytes(b"checkpoint")
    destination = tmp_path / "local" / "checkpoint.pt"
    destination.parent.mkdir()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setattr("core.models.factory.hf_hub_download", download)
    result = ModelFactory._ensure_local("owner/model", "checkpoint.pt", str(destination))

    assert result == str(destination)
    assert destination.read_bytes() == b"checkpoint"
    assert calls == [
        {
            "repo_id": "owner/model",
            "filename": "checkpoint.pt",
            "repo_type": "model",
            "token": None,
        }
    ]

    assert ModelFactory._ensure_local("owner/model", "checkpoint.pt", str(destination)) == str(destination)
    assert len(calls) == 1


def test_model_factory_returns_cached_model_without_loading(tmp_path):
    model = object()
    ModelFactory._cache["image"] = {"model": model}

    assert ModelFactory.get_model("image") is model
    assert ModelFactory.get_model("unknown") is None
