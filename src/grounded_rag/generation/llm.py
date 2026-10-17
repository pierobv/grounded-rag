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


class LLM(Protocol):
    @property
    def name(self) -> str: ...

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[ChatEvent]: ...


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

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[ChatEvent]:
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
                    raise OllamaError(
                        f"Ollama respondeu {response.status_code}: {response.text[:200]}"
                    )

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
