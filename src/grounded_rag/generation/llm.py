"""Cliente de chat do Ollama, com streaming.

    POST {OLLAMA_BASE_URL}/api/chat
    {"model": ..., "messages": [...], "stream": true, "options": {...}}

Com stream=true, a resposta é NDJSON: um objeto JSON por linha, à medida que o modelo gera.
    {"message": {"role": "assistant", "content": "Use"}, "done": false}
    {"message": {"role": "assistant", "content": " the"}, "done": false}
    ...
    {"done": true, "prompt_eval_count": 1234, "eval_count": 87, ...}
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from grounded_rag.retrieval.embeddings import OllamaError


@dataclass(frozen=True)
class ChatDelta:
    """Pedaço de texto gerado."""

    text: str


@dataclass(frozen=True)
class ChatDone:
    """Fim da geração, com as contagens que o Ollama devolve."""

    prompt_tokens: int
    completion_tokens: int
    context_window: int  # num_ctx usado, para detectar prompts perto do limite


ChatEvent = ChatDelta | ChatDone

# Erros 5xx do Ollama costumam ser transitórios (ex.: "timed out waiting for llama-server to
# start" quando a memória está apertada e ele está trocando de modelo). Tentamos de novo.
RETRIES = 3
RETRY_WAIT_S = 10.0


class LLM(Protocol):
    @property
    def name(self) -> str: ...

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[ChatEvent]: ...


class _TransientError(Exception):
    """Erro 5xx antes do streaming começar: vale tentar de novo."""


class OllamaChat:
    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int = 8192,
        temperature: float = 0.0,
        seed: int = 42,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,  # injetado nos testes (MockTransport)
    ) -> None:
        self.model = model
        # Opções do modelo enviadas ao Ollama:
        # - num_ctx: tamanho do contexto em tokens. Se o prompt passar disso, o Ollama CORTA
        #   o começo do prompt em silêncio (os trechos sumiriam sem erro). Por isso é explícito.
        # - temperature 0 + seed fixa: respostas reprodutíveis, essencial para a avaliação.
        self.options = {"num_ctx": num_ctx, "temperature": temperature, "seed": seed}
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout, transport=transport)

    @property
    def name(self) -> str:
        return self.model

    def complete_json(self, messages: list[dict[str, str]], schema: dict) -> dict:
        """Resposta inteira (sem streaming) forçada a seguir um JSON Schema.

        O Ollama aceita o schema no campo `format` e restringe a geração token a token
        (structured outputs): a saída é sempre JSON válido nesse formato.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": self.options,
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            for attempt in range(1, RETRIES):
                if response.status_code < 500:
                    break
                time.sleep(RETRY_WAIT_S * attempt)
                response = self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaError(f"Não foi possível conectar ao Ollama em {self._base_url}.") from exc
        if response.status_code == 404:
            raise OllamaError(
                f"Modelo '{self.model}' não encontrado. Rode: ollama pull {self.model}"
            )
        if response.is_error:
            raise OllamaError(f"Ollama respondeu {response.status_code}: {response.text[:200]}")
        return json.loads(response.json()["message"]["content"])

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[ChatEvent]:
        # Tenta de novo só se o erro 5xx vier ANTES do primeiro pedaço de texto: depois que o
        # streaming começou, repetir duplicaria o texto já entregue.
        for attempt in range(1, RETRIES + 1):
            try:
                yield from self._stream_chat_once(messages)
                return
            except _TransientError as exc:
                if attempt == RETRIES:
                    raise OllamaError(str(exc)) from exc
                time.sleep(RETRY_WAIT_S * attempt)

    def _stream_chat_once(self, messages: list[dict[str, str]]) -> Iterator[ChatEvent]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options,
        }
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code == 404:
                    raise OllamaError(
                        f"Modelo '{self.model}' não encontrado no Ollama. "
                        f"Rode: ollama pull {self.model}"
                    )
                if response.is_error:
                    response.read()
                    message = f"Ollama respondeu {response.status_code}: {response.text[:200]}"
                    if response.status_code >= 500:
                        raise _TransientError(message)
                    raise OllamaError(message)

                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:  # erro no meio do streaming
                        raise OllamaError(f"Ollama: {event['error']}")
                    if event.get("done"):
                        yield ChatDone(
                            prompt_tokens=event.get("prompt_eval_count", 0),
                            completion_tokens=event.get("eval_count", 0),
                            context_window=self.options["num_ctx"],
                        )
                        return
                    text = event.get("message", {}).get("content", "")
                    if text:
                        yield ChatDelta(text)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Não foi possível conectar ao Ollama em {self._base_url}. "
                "Ele está rodando? Confira OLLAMA_BASE_URL no .env."
            ) from exc
