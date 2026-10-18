import json

import httpx
import pytest

from grounded_rag.generation.llm import ChatDelta, ChatDone, OllamaChat
from grounded_rag.retrieval.embeddings import OllamaError

MESSAGES = [{"role": "user", "content": "oi"}]


def ndjson(*events) -> bytes:
    """Corpo de resposta em NDJSON: um objeto JSON por linha, como o Ollama transmite."""
    return "\n".join(json.dumps(e) for e in events).encode()


def make_llm(handler, **kwargs) -> OllamaChat:
    return OllamaChat(
        base_url="http://ollama.test:11434",
        model="qwen2.5:7b",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_streams_deltas_then_done_with_counts():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        body = ndjson(
            {"message": {"role": "assistant", "content": "Use "}, "done": False},
            {"message": {"role": "assistant", "content": "HttpOnly [1]."}, "done": False},
            {"done": True, "prompt_eval_count": 812, "eval_count": 7},
        )
        return httpx.Response(200, content=body)

    events = list(make_llm(handler, num_ctx=4096).stream_chat(MESSAGES))

    assert events == [
        ChatDelta("Use "),
        ChatDelta("HttpOnly [1]."),
        ChatDone(prompt_tokens=812, completion_tokens=7, context_window=4096),
    ]
    body = requests[0]
    assert body["stream"] is True
    assert body["messages"] == MESSAGES
    assert body["options"] == {"num_ctx": 4096, "temperature": 0.0, "seed": 42}


def test_missing_model_suggests_pull():
    def handler(request):
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(OllamaError, match="ollama pull qwen2.5:7b"):
        list(make_llm(handler).stream_chat(MESSAGES))


def test_error_in_the_middle_of_the_stream():
    def handler(request):
        body = ndjson(
            {"message": {"content": "Use"}, "done": False},
            {"error": "model runner crashed"},
        )
        return httpx.Response(200, content=body)

    with pytest.raises(OllamaError, match="runner crashed"):
        list(make_llm(handler).stream_chat(MESSAGES))


def test_connection_error_has_helpful_message():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="OLLAMA_BASE_URL"):
        list(make_llm(handler).stream_chat(MESSAGES))


def test_transient_5xx_is_retried(monkeypatch):
    monkeypatch.setattr("grounded_rag.generation.llm.RETRY_WAIT_S", 0)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500, json={"error": "timed out waiting for llama-server"})
        return httpx.Response(
            200, content=ndjson({"message": {"content": "ok"}, "done": False}, {"done": True})
        )

    events = list(make_llm(handler).stream_chat(MESSAGES))

    assert len(calls) == 2
    assert events[0] == ChatDelta("ok")


def test_persistent_5xx_gives_up(monkeypatch):
    monkeypatch.setattr("grounded_rag.generation.llm.RETRY_WAIT_S", 0)

    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    with pytest.raises(OllamaError, match="500"):
        list(make_llm(handler).stream_chat(MESSAGES))


def test_complete_json_retries_and_parses(monkeypatch):
    monkeypatch.setattr("grounded_rag.generation.llm.RETRY_WAIT_S", 0)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"message": {"content": '{"supported": true}'}})

    result = make_llm(handler).complete_json(MESSAGES, {"type": "object"})

    assert result == {"supported": True}
    assert calls[1]["format"] == {"type": "object"} and calls[1]["stream"] is False
