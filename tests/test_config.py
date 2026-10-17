import pytest
from pydantic import ValidationError

from grounded_rag.config import Settings

# _env_file=None: os testes ignoram qualquer .env local da máquina,
# para que o resultado dependa só do que o próprio teste define.


def test_reads_ollama_url_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    settings = Settings(_env_file=None)

    assert str(settings.ollama_base_url) == "http://localhost:11434/"


def test_missing_ollama_url_fails_clearly(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    with pytest.raises(ValidationError, match="ollama_base_url"):
        Settings(_env_file=None)


def test_invalid_ollama_url_is_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "nao-e-uma-url")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_models_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")

    settings = Settings(_env_file=None)

    assert settings.llm_model == "llama3.1:8b"
