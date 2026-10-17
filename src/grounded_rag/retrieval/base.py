"""Tipos comuns a todos os buscadores (vetorial, BM25 e, na etapa 3, híbrido)."""

from dataclasses import dataclass, field
from typing import Protocol

from grounded_rag.ingest.models import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float  # escala depende do buscador: similaridade de cosseno ou score BM25
    rank: int  # posição na lista, a partir de 1
    # De onde o resultado veio: {"vector": 3, "bm25": 1} = 3º no vetorial, 1º no BM25.
    # Preenchido pela busca híbrida; serve para explicar cada resultado.
    origins: dict[str, int] = field(default_factory=dict)


class Retriever(Protocol):
    """Qualquer objeto com este método é um buscador (tipagem estrutural, como "duck typing").

    É a interface que a fusão RRF vai combinar: ela só precisa dos rankings.
    """

    def search(self, query: str, k: int) -> list[SearchResult]: ...


def to_results(scored: list[tuple[Chunk, float]]) -> list[SearchResult]:
    """Converte pares (trecho, score), já ordenados, em resultados numerados."""
    return [SearchResult(chunk, score, rank) for rank, (chunk, score) in enumerate(scored, 1)]
