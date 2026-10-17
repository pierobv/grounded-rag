"""Testes contra o Ollama real. Fora do CI; rode localmente com: uv run pytest -m ollama"""

import math

import pytest

from grounded_rag.config import Settings
from grounded_rag.retrieval.embeddings import OllamaEmbedder

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def embedder():
    settings = Settings()  # lê o .env local (OLLAMA_BASE_URL, modelo, prefixos)
    return OllamaEmbedder(
        base_url=str(settings.ollama_base_url),
        model=settings.embedding_model,
        doc_prefix=settings.embed_doc_prefix,
        query_prefix=settings.embed_query_prefix,
    )


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_embeddings_have_consistent_dimension(embedder):
    vectors = embedder.embed_documents(["first text", "second text"])

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0


def test_related_text_is_closer_than_unrelated(embedder):
    query = embedder.embed_query("How do I prevent SQL injection?")
    related, unrelated = embedder.embed_documents(
        [
            "Use parameterized queries so user input is never concatenated into SQL.",
            "Configure the Strict-Transport-Security header to enforce HTTPS.",
        ]
    )

    assert cosine(query, related) > cosine(query, unrelated)
