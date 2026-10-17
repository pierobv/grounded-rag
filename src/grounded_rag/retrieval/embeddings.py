"""Embeddings via API HTTP do Ollama.

Chamamos a API diretamente com httpx (sem SDK) para que fique explícito o que é
enviado e recebido:

    POST {OLLAMA_BASE_URL}/api/embed
    {"model": "nomic-embed-text", "input": ["texto 1", "texto 2", ...]}
    -> {"embeddings": [[0.01, ...], [0.03, ...]]}
"""

from typing import Protocol

import httpx

Vector = list[float]


class Embedder(Protocol):
    """Interface usada pelo índice vetorial. Nos testes, um embedder falso a implementa."""

    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class OllamaError(RuntimeError):
    """Falha ao falar com o Ollama, com mensagem pensada para quem roda o CLI."""


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        doc_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 32,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,  # injetado nos testes (MockTransport)
    ) -> None:
        self.model = model
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout, transport=transport)

    @property
    def name(self) -> str:
        return self.model

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        # Modelos como o nomic-embed-text foram treinados com um prefixo que diz a tarefa:
        # "search_document: " para o que será buscado, "search_query: " para a pergunta.
        prefixed = [self.doc_prefix + t for t in texts]
        vectors: list[Vector] = []
        # Lotes: uma requisição por texto seria lenta; uma só com milhares, pesada demais.
        for start in range(0, len(prefixed), self.batch_size):
            vectors.extend(self._embed(prefixed[start : start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self._embed([self.query_prefix + text])[0]

    def _embed(self, inputs: list[str]) -> list[Vector]:
        try:
            response = self._client.post("/api/embed", json={"model": self.model, "input": inputs})
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Não foi possível conectar ao Ollama em {self._base_url}. "
                "Ele está rodando? Confira OLLAMA_BASE_URL no .env."
            ) from exc

        if response.status_code == 404:
            raise OllamaError(
                f"Modelo '{self.model}' não encontrado no Ollama. Rode: ollama pull {self.model}"
            )
        if response.is_error:
            raise OllamaError(f"Ollama respondeu {response.status_code}: {response.text[:200]}")

        embeddings = response.json()["embeddings"]
        if len(embeddings) != len(inputs):  # proteção: nunca associar vetor ao trecho errado
            raise OllamaError(f"Esperava {len(inputs)} embeddings, recebi {len(embeddings)}")
        return embeddings
