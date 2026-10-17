"""Busca híbrida: funde rankings de vários buscadores com Reciprocal Rank Fusion (RRF).

    score_rrf(trecho) = Σ  1 / (k + posição_i(trecho))
                        i = buscadores em que o trecho aparece

Por que posições e não scores? Cada buscador tem sua escala (cosseno ~0,5–0,8; BM25 ~6–24).
Somar scores exigiria normalizá-los, e qualquer normalização é frágil. O RRF ignora os scores
e usa só a posição: estar no topo de uma lista vale muito, estar no topo das duas vale mais.

k (padrão 60, do artigo original de Cormack et al., 2009) suaviza a diferença entre posições:
com k=60, a 1ª posição vale 1/61 ≈ 0,0164 e a 10ª vale 1/70 ≈ 0,0143. Um trecho que aparece
em 5º nas duas listas (2/65 ≈ 0,0308) supera um que é 1º em só uma (0,0164).
"""

from grounded_rag.ingest.models import Chunk
from grounded_rag.retrieval.base import Retriever, SearchResult


def rrf(rankings: dict[str, list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Funde rankings nomeados ({"vector": [...], "bm25": [...]}) em um só.

    Trechos são identificados pelo ID, então o mesmo trecho vindo de duas listas
    soma as duas contribuições. Empates são resolvidos pelo ID (resultado determinístico).
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    origins: dict[str, dict[str, int]] = {}

    for name, results in rankings.items():
        for result in results:
            chunk_id = result.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + result.rank)
            chunks[chunk_id] = result.chunk
            origins.setdefault(chunk_id, {})[name] = result.rank

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        SearchResult(chunks[chunk_id], scores[chunk_id], rank, origins[chunk_id])
        for rank, chunk_id in enumerate(ordered, start=1)
    ]


class HybridRetriever:
    def __init__(self, retrievers: dict[str, Retriever], candidates: int, k: int = 60) -> None:
        self.retrievers = retrievers
        self.candidates = candidates  # quantos resultados pedir a CADA buscador
        self.k = k

    def search(self, query: str, k: int) -> list[SearchResult]:
        # Pedimos mais que k a cada buscador: um trecho em 15º nas duas listas pode
        # terminar no top 5 da fusão, e ele só entra na conta se for buscado.
        n = max(k, self.candidates)
        rankings = {name: r.search(query, n) for name, r in self.retrievers.items()}
        return rrf(rankings, self.k)[:k]
