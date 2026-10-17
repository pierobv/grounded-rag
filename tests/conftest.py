"""Utilitários compartilhados pelos testes (o pytest carrega este arquivo automaticamente)."""

import hashlib
import math

import pytest

from grounded_rag.retrieval.bm25 import tokenize

DIMS = 64


class FakeEmbedder:
    """Embedder determinístico e offline: substitui o Ollama nos testes.

    Cada palavra cai em uma das 64 posições do vetor (via hash). Textos que
    compartilham palavras ficam com vetores parecidos, o que basta para testar
    a busca vetorial sem nenhum modelo. Também conta quantos textos recebeu.
    """

    name = "fake-embedder"

    def __init__(self) -> None:
        self.embedded_texts = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMS
        for token in tokenize(text):
            # hashlib, não hash(): o hash() do Python muda a cada execução.
            slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS
            vector[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


class FakeReranker:
    """Reranker offline: nota = quantas palavras da pergunta aparecem no texto."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        terms = set(tokenize(query))
        return [float(len(terms & set(tokenize(t)))) for t in texts]


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()
