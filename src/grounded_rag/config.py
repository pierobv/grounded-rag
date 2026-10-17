"""Configuração central do projeto.

Toda configuração vem de variáveis de ambiente (ou de um arquivo `.env` local,
ignorado pelo Git). Nada de endereços, chaves ou caminhos pessoais no código.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import AnyHttpUrl, PositiveInt, model_validator
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
    processed_dir: Path = Path("data/processed")
    chroma_dir: Path = Path(".chroma")

    # Chunking (tamanho em palavras). Valores iniciais, não ajustados: a etapa de
    # avaliação mede o efeito de cada combinação.
    chunk_strategy: Literal["fixed", "structure"] = "structure"
    chunk_size: PositiveInt = 200
    chunk_overlap: int = 40

    # Embeddings. O nomic-embed-text foi treinado com prefixos de tarefa; outro modelo
    # pode precisar de outros prefixos (ou nenhum: defina a variável vazia).
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    embed_batch_size: PositiveInt = 32

    # Busca
    retrieval_mode: Literal["vector", "bm25"] = "vector"
    top_k: PositiveInt = 5

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        # Validação entre campos: roda depois que cada campo já foi validado sozinho.
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP deve estar entre 0 e CHUNK_SIZE - 1")
        return self


@lru_cache
def get_settings() -> Settings:
    """Retorna as configurações, lidas uma única vez (cache).

    Usar uma função em vez de uma variável global facilita os testes:
    basta limpar o cache (`get_settings.cache_clear()`) e mudar o ambiente.
    """
    return Settings()
