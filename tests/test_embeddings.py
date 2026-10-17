import json

import httpx
import pytest

from grounded_rag.retrieval.embeddings import OllamaEmbedder, OllamaError

# httpx.MockTransport troca a rede por uma função nossa: o teste vê cada requisição
# e decide a resposta. Nenhum Ollama de verdade é necessário (o CI não tem modelo).


def make_embedder(handler, **kwargs) -> OllamaEmbedder:
    defaults = {"base_url": "http://ollama.test:11434", "model": "nomic-embed-text"}
    return OllamaEmbedder(**(defaults | kwargs), transport=httpx.MockTransport(handler))


def echo_handler(requests: list):
    """Responde um vetor [tamanho do texto] por entrada e guarda o corpo de cada requisição."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        return httpx.Response(200, json={"embeddings": [[len(t)] for t in body["input"]]})

    return handler


def test_sends_model_and_document_prefix():
    requests = []
    embedder = make_embedder(echo_handler(requests), doc_prefix="search_document: ")

    embedder.embed_documents(["xss"])

    path, body = requests[0]
    assert path == "/api/embed"
    assert body == {"model": "nomic-embed-text", "input": ["search_document: xss"]}


def test_query_uses_query_prefix():
    requests = []
    embedder = make_embedder(echo_handler(requests), query_prefix="search_query: ")

    embedder.embed_query("o que é csrf?")

    assert requests[0][1]["input"] == ["search_query: o que é csrf?"]


def test_documents_are_sent_in_batches_and_keep_order():
    requests = []
    embedder = make_embedder(echo_handler(requests), batch_size=2)

    vectors = embedder.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert [len(body["input"]) for _, body in requests] == [2, 2, 1]
    assert vectors == [[1], [2], [3], [4], [5]]


def test_connection_error_has_helpful_message():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(OllamaError, match="OLLAMA_BASE_URL"):
        make_embedder(handler).embed_query("x")


def test_missing_model_suggests_pull():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'x' not found"})

    with pytest.raises(OllamaError, match="ollama pull nomic-embed-text"):
        make_embedder(handler).embed_query("x")


def test_wrong_number_of_embeddings_is_an_error():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    with pytest.raises(OllamaError, match="Esperava 2"):
        make_embedder(handler).embed_documents(["a", "b"])
