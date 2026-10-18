"""Reranking com cross-encoder: reordena os candidatos de outro buscador.

Bi-encoder (busca vetorial): pergunta e trecho viram vetores SEPARADAMENTE, e comparamos os
vetores. Rápido (os trechos são embeddados uma vez só), mas a pergunta nunca "vê" o trecho.

Cross-encoder: pergunta e trecho entram JUNTOS no modelo, que dá uma nota de relevância.
Bem mais preciso, mas exige uma passada do modelo por par (pergunta, trecho): inviável para o
corpus inteiro. Por isso o fluxo é: busca barata traz N candidatos -> cross-encoder reordena.
"""

from functools import lru_cache
from typing import Protocol

from grounded_rag.retrieval.base import Retriever, SearchResult


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Carrega o modelo uma vez por processo (a avaliação roda vários experimentos seguidos)."""
    # Import aqui dentro: carregar sentence-transformers traz o PyTorch (segundos de
    # inicialização). Assim, quem não usa reranking (e os testes) não paga esse custo.
    from sentence_transformers import CrossEncoder

    # Na primeira execução o modelo é baixado do Hugging Face e fica em cache local.
    # O dispositivo é escolhido automaticamente (MPS = GPU do Mac, quando disponível).
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model = _load_cross_encoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict([(query, text) for text in texts])
        return [float(s) for s in scores]


class RerankingRetriever:
    """Envolve qualquer buscador: busca `candidates` trechos e devolve os k mais bem avaliados."""

    def __init__(self, base: Retriever, reranker: Reranker, candidates: int) -> None:
        self.base = base
        self.reranker = reranker
        self.candidates = candidates

    def search(self, query: str, k: int) -> list[SearchResult]:
        candidates = self.base.search(query, max(k, self.candidates))
        scores = self.reranker.score(query, [c.chunk.text for c in candidates])

        # sorted é estável: em empate de score, vale a ordem da busca original.
        reranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: -pair[1])
        return [
            SearchResult(
                chunk=result.chunk,
                score=score,
                rank=rank,
                # Mantém a origem e registra a posição antes do reranking.
                origins={**result.origins, "before_rerank": result.rank},
            )
            for rank, (result, score) in enumerate(reranked[:k], start=1)
        ]
