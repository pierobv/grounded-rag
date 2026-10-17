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


class FakeLLM:
    """LLM offline: devolve um texto fixo em pedaços, como se fosse streaming."""

    name = "fake-llm"

    def __init__(self, reply: str = "Use HttpOnly cookies [1].", piece: int = 4) -> None:
        self.reply = reply
        self.piece = piece
        self.calls: list[list[dict[str, str]]] = []

    def stream_chat(self, messages):
        from grounded_rag.generation.llm import ChatDelta, ChatDone

        self.calls.append(messages)
        for start in range(0, len(self.reply), self.piece):
            yield ChatDelta(self.reply[start : start + self.piece])
        yield ChatDone(prompt_tokens=100, completion_tokens=10, context_window=8192)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
