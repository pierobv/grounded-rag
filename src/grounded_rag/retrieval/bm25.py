"""Busca lexical com BM25: pontua trechos pelos termos exatos da consulta.

BM25, em resumo: um termo pesa mais quando
- aparece mais vezes no trecho (com retorno decrescente, controlado por k1),
- é raro no corpus (IDF: "csrf" vale mais que "the"),
- e o trecho não é longo demais (normalização por tamanho, controlada por b).

O índice é montado em memória a partir dos trechos: para alguns milhares de trechos
isso leva menos de um segundo, então não vale a complexidade de persistir em disco.
"""

import re

from rank_bm25 import BM25Okapi

from grounded_rag.ingest.models import Chunk
from grounded_rag.retrieval.base import SearchResult, to_results

# [^\W_]: qualquer letra ou dígito Unicode (inclui acentos: "saída" é UM token).
# Um [a-z0-9]+ quebraria "saída" em "sa" + "da", ruim para material em português.
TOKEN_RE = re.compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    """Minúsculas + sequências de letras/números. "X-Frame-Options" -> ["x", "frame", "options"].

    lower() em vez de casefold() é suficiente para inglês e português.

    Propositalmente simples. Stopwords, stemming e preservar termos com hífen são
    experimentos para a etapa de avaliação.
    """
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        # BM25Okapi recebe o corpus já tokenizado; k1=1.5 e b=0.75 são os padrões da biblioteca.
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[SearchResult]:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Índices ordenados pelo score, do maior para o menor; desempate pela ordem original.
        best = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        # Score 0 = nenhum termo da consulta no trecho: não é um resultado.
        return to_results([(self._chunks[i], float(scores[i])) for i in best if scores[i] > 0])
