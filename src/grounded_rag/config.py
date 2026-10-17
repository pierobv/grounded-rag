"""Configuração central do projeto.

Toda configuração vem de variáveis de ambiente (ou de um arquivo `.env` local,
ignorado pelo Git). Nada de endereços, chaves ou caminhos pessoais no código.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Lê variáveis do ambiente e, se existir, do arquivo .env.
    # O nome da variável é o nome do campo em maiúsculas: ollama_base_url -> OLLAMA_BASE_URL.
    # extra="ignore": variáveis desconhecidas no .env não quebram a aplicação.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Obrigatório e sem valor padrão: o endereço do Ollama muda entre máquinas
    # (localhost no Mac, PC via Tailscale), então nunca fica fixo no código.
    ollama_base_url: AnyHttpUrl

    # Modelos servidos pelo Ollama. São só nomes públicos de modelos, não segredos,
    # por isso têm padrão; podem ser trocados por variável de ambiente.
    llm_model: str = "qwen2.5:7b"
    dev_llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"

    # Caminhos de dados (relativos à raiz do projeto).
    data_dir: Path = Path("data/public")
    chroma_dir: Path = Path(".chroma")


@lru_cache
def get_settings() -> Settings:
    """Retorna as configurações, lidas uma única vez (cache).

    Usar uma função em vez de uma variável global facilita os testes:
    basta limpar o cache (`get_settings.cache_clear()`) e mudar o ambiente.
    """
    return Settings()
