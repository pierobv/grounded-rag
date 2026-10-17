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


@pytest.fixture(scope="module")
def answer():
    """Responde com o modelo pequeno (DEV_LLM_MODEL) usando trechos fixos."""
    from grounded_rag.generation.answer import Final, stream_answer
    from grounded_rag.generation.llm import OllamaChat
    from grounded_rag.ingest.models import Chunk
    from grounded_rag.retrieval.base import to_results

    settings = Settings()
    chunks = [
        Chunk(
            id="1",
            source="owasp/Session.md",
            location="Cookies",
            text="Set the HttpOnly attribute on session cookies so JavaScript cannot read them.",
            strategy="structure",
        )
    ]

    class Fixed:
        def search(self, query, k):
            return to_results([(c, 1.0) for c in chunks])

    llm = OllamaChat(str(settings.ollama_base_url), settings.dev_llm_model)

    def ask(question):
        events = list(stream_answer(question, Fixed(), llm, k=5))
        assert isinstance(events[-1], Final)
        return events[-1].answer

    return ask


def test_answerable_question_is_cited(answer):
    result = answer("How do I stop JavaScript from reading the session cookie?")

    assert not result.abstained
    assert [c.number for c in result.citations] == [1]


def test_out_of_corpus_question_abstains(answer):
    assert answer("What is the capital of France?").abstained
