"""Transforma o texto gerado em uma resposta validada (Pydantic), com citações conferidas.

Aqui o Pydantic faz sentido (e não dataclass): o texto vem do LLM, ou seja, é entrada
NÃO confiável. As regras de "resposta fundamentada" ficam nos validadores do modelo,
então nenhuma resposta inválida consegue sequer ser construída.
"""

import re
from typing import Self

from pydantic import BaseModel, model_validator

from grounded_rag.generation.prompt import SENTINEL_RE
from grounded_rag.ingest.models import Chunk

# [1]   [1][3]   [1, 3]   [1,3]
MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

DEFAULT_ABSTAIN_MESSAGE = "Não encontrei essa informação nos documentos."


class Citation(BaseModel):
    number: int  # número do trecho no prompt ([1] = primeiro trecho)
    source: str
    location: str


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    abstained: bool
    # Marcadores que apontavam para trechos inexistentes (alucinação de citação).
    # Foram removidos do texto, mas ficam registrados: a avaliação vai contá-los.
    invalid_citations: list[int] = []
    raw_text: str  # texto original do modelo, antes de qualquer limpeza (para depuração)

    @model_validator(mode="after")
    def _grounded(self) -> Self:
        if not self.abstained and not self.citations:
            raise ValueError("Resposta sem abstenção precisa de pelo menos uma citação válida")
        numbers = [c.number for c in self.citations]
        if len(numbers) != len(set(numbers)):
            raise ValueError("Citações repetidas")
        return self


def abstention(raw_text: str, message: str = DEFAULT_ABSTAIN_MESSAGE) -> Answer:
    return Answer(text=message, citations=[], abstained=True, raw_text=raw_text)


def build_answer(raw_text: str, chunks: list[Chunk]) -> Answer:
    """Valida o texto do modelo contra os trechos que foram enviados no prompt.

    1. Sentinela presente -> abstenção (usando a frase que o modelo escreveu depois dele).
    2. Marcadores para trechos inexistentes ([7] com 5 trechos) são removidos e registrados.
    3. Nenhuma citação válida -> abstenção: afirmação sem fonte não chega ao usuário.
    """
    text = raw_text.strip()

    if match := SENTINEL_RE.search(text):
        message = text[match.end() :].strip(" :.-\n")
        message = MARKER_RE.sub("", message).strip()  # ex.: "INSUFFICIENT_CONTEXT[1]"
        return abstention(raw_text, f"{message}." if message else DEFAULT_ABSTAIN_MESSAGE)

    valid: list[int] = []  # na ordem da primeira aparição no texto
    invalid: list[int] = []

    def check_marker(match: re.Match) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        kept = [n for n in numbers if 1 <= n <= len(chunks)]
        for n in numbers:
            target = valid if n in kept else invalid
            if n not in target:
                target.append(n)
        # Reescreve o marcador só com os números válidos ("[2, 9]" -> "[2]"; "[9]" -> "").
        return "".join(f"[{n}]" for n in kept)

    cleaned = MARKER_RE.sub(check_marker, text)

    if not valid:
        return abstention(raw_text)

    citations = [
        Citation(number=n, source=chunks[n - 1].source, location=chunks[n - 1].location)
        for n in valid
    ]
    return Answer(
        text=cleaned,
        citations=citations,
        abstained=False,
        invalid_citations=invalid,
        raw_text=raw_text,
    )
