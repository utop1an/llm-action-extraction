import sys
from types import SimpleNamespace

from src.llm.ollama import OllamaClient


def test_ollama_client_uses_env_host(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, host):
            captured["client_host"] = host

    class FakeAsyncClient:
        def __init__(self, host):
            captured["async_client_host"] = host

    monkeypatch.setitem(
        sys.modules,
        "ollama",
        SimpleNamespace(Client=FakeClient, AsyncClient=FakeAsyncClient),
    )
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:23456")

    OllamaClient({"model_name": "llama3.3:70b"})

    assert captured["client_host"] == "http://127.0.0.1:23456"
    assert captured["async_client_host"] == "http://127.0.0.1:23456"
