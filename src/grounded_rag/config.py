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
    # bge-m3: multilíngue, venceu a avaliação (docs/decisions.md, decisões 029 e 030).
    embedding_model: str = "bge-m3"

    # Caminhos de dados (relativos à raiz do projeto).
    data_dir: Path = Path("data/public")
    processed_dir: Path = Path("data/processed")
    chroma_dir: Path = Path(".chroma")

    # Chunking (tamanho em palavras). Valores iniciais, não ajustados: a etapa de
    # avaliação mede o efeito de cada combinação.
    chunk_strategy: Literal["fixed", "structure"] = "structure"
    chunk_size: PositiveInt = 200
    chunk_overlap: int = 40
    # Contextual chunk headers: o caminho de títulos vai no início do texto de cada trecho.
    chunk_headings: bool = True

    # Prefixos de tarefa do embedding. O bge-m3 não usa; o nomic-embed-text usa
    # "search_document: " e "search_query: ".
    embed_doc_prefix: str = ""
    embed_query_prefix: str = ""
    embed_batch_size: PositiveInt = 32

    # Busca
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    top_k: PositiveInt = 5
    # Quantos candidatos cada buscador fornece para a fusão e para o reranking.
    candidates: PositiveInt = 20
    rrf_k: PositiveInt = 60

    # Reranking com cross-encoder (sentence-transformers; baixado do Hugging Face no 1º uso).
    rerank: bool = True
    # bge-reranker-v2-m3: multilíngue e mais preciso; ~2–4 s por pergunta no M4.
    # Alternativa rápida (só inglês): cross-encoder/ms-marco-MiniLM-L-6-v2.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # Geração
    num_ctx: PositiveInt = 8192  # contexto do LLM em tokens (o Ollama corta o que passar)
    temperature: float = 0.0
    seed: int = 42
    # Portão de confiança: se o melhor trecho (após reranking) tiver score abaixo disso,
    # responde "não sei" sem chamar o LLM. Desligado (vazio) até ser calibrado na avaliação.
    min_rerank_score: float | None = None

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
