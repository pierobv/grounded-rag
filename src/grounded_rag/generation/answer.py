"""Orquestração: buscar trechos -> montar prompt -> gerar com streaming -> validar citações.

`stream_answer` produz eventos, na ordem:
    Retrieved   (os trechos que vão no prompt)
    TextDelta*  (pedaços de texto, para mostrar ao vivo)
    Final       (resposta validada + métricas)
O CLI usa agora; a API (/ask com streaming) e a avaliação vão usar a mesma função.
"""

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

from grounded_rag.generation.citations import Answer, abstention, build_answer
from grounded_rag.generation.llm import LLM, ChatDelta, ChatDone
from grounded_rag.generation.prompt import ABSTAIN_SENTINEL, PROMPT_VERSION, build_messages
from grounded_rag.retrieval.base import Retriever, SearchResult

# Acima desta fração do contexto, o prompt corre risco de ser cortado pelo Ollama.
CONTEXT_WARNING_RATIO = 0.9


@dataclass(frozen=True)
class GenerationStats:
    model: str
    prompt_version: str
    retrieval_ms: float
    time_to_first_token_ms: float | None  # None quando o LLM não foi chamado
    total_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_near_limit: bool = False

    @property
    def tokens_per_second(self) -> float:
        generation_s = (self.total_ms - (self.time_to_first_token_ms or 0)) / 1000
        return self.completion_tokens / generation_s if generation_s > 0 else 0.0


@dataclass(frozen=True)
class Retrieved:
    results: list[SearchResult]


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Final:
    answer: Answer
    stats: GenerationStats


AnswerEvent = Retrieved | TextDelta | Final


class _SentinelFilter:
    """Segura o início do texto até saber se a resposta começa com o sentinela.

    Sem isso, o usuário veria "INSUFFICIENT_CONTEXT" aparecer na tela durante o streaming.
    Assim que o texto deixa de poder ser o começo do sentinela, tudo é liberado.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._decided = False
        self._suppress = False

    def feed(self, text: str) -> str:
        if self._decided:
            return "" if self._suppress else text
        self._pending += text
        # Normaliza como a detecção tolerante: "Insufficient Context" -> "INSUFFICIENT_CONTEXT".
        head = re.sub(r"[\s-]", "_", self._pending.lstrip().upper())
        if head and not ABSTAIN_SENTINEL.startswith(head[: len(ABSTAIN_SENTINEL)]):
            self._decided = True  # não é o sentinela: libera o que estava guardado
            return self._pending
        if len(head) >= len(ABSTAIN_SENTINEL):
            self._decided = self._suppress = True  # começa com o sentinela: esconde
        return ""

    def flush(self) -> str:
        return "" if self._decided else self._pending


def stream_answer(
    question: str,
    retriever: Retriever,
    llm: LLM,
    k: int,
    min_score: float | None = None,
) -> Iterator[AnswerEvent]:
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - started) * 1000

    results = retriever.search(question, k)
    retrieval_ms = elapsed_ms()
    yield Retrieved(results)

    # Portão de confiança: se nada foi encontrado, ou o melhor trecho tem score baixo,
    # responde "não sei" sem gastar uma chamada ao LLM.
    if not results or (min_score is not None and results[0].score < min_score):
        stats = GenerationStats(llm.name, PROMPT_VERSION, retrieval_ms, None, elapsed_ms())
        yield Final(abstention(raw_text=""), stats)
        return

    chunks = [r.chunk for r in results]
    text_parts: list[str] = []
    first_token_ms: float | None = None
    done: ChatDone | None = None
    sentinel = _SentinelFilter()

    for event in llm.stream_chat(build_messages(question, chunks)):
        if isinstance(event, ChatDelta):
            if first_token_ms is None:
                first_token_ms = elapsed_ms()
            text_parts.append(event.text)
            if visible := sentinel.feed(event.text):
                yield TextDelta(visible)
        else:
            done = event
    if tail := sentinel.flush():
        yield TextDelta(tail)

    prompt_tokens = done.prompt_tokens if done else 0
    stats = GenerationStats(
        model=llm.name,
        prompt_version=PROMPT_VERSION,
        retrieval_ms=retrieval_ms,
        time_to_first_token_ms=first_token_ms,
        total_ms=elapsed_ms(),
        prompt_tokens=prompt_tokens,
        completion_tokens=done.completion_tokens if done else 0,
        context_near_limit=bool(
            done and prompt_tokens >= CONTEXT_WARNING_RATIO * done.context_window
        ),
    )
    yield Final(build_answer("".join(text_parts), chunks), stats)
