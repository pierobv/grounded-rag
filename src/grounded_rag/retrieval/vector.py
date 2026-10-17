"""Busca vetorial: embeddings do Ollama guardados no Chroma (modo persistente, sem servidor).

O Chroma aqui é só armazenamento + busca de vizinhos mais próximos (HNSW).
Os embeddings são calculados por nós (embedding_function=None) e entregues prontos:
assim o modelo é sempre o do Ollama e o Chroma não baixa um modelo próprio escondido.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

from grounded_rag.ingest.models import Chunk
from grounded_rag.retrieval.base import SearchResult, to_results
from grounded_rag.retrieval.embeddings import Embedder

# Nome de coleção no Chroma: 3–512 caracteres de [a-zA-Z0-9._-], começando e terminando
# com letra ou número.
INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def collection_name(index_name: str, embedding_model: str) -> str:
    """Uma coleção por configuração (chunking + modelo), ex.: structure-200-40-nomic-embed-text.

    Trocar a estratégia ou o modelo gera outra coleção, então índices de experimentos
    diferentes nunca se misturam e podem ser comparados lado a lado.
    """
    raw = f"{index_name}-{embedding_model.removesuffix(':latest')}"
    return INVALID_NAME_CHARS.sub("-", raw).strip("-._")


@dataclass(frozen=True)
class SyncStats:
    added: int
    removed: int
    kept: int


class VectorIndex:
    def __init__(self, chroma_dir: Path, name: str, embedder: Embedder) -> None:
        self.embedder = embedder
        client = chromadb.PersistentClient(
            path=str(chroma_dir), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        # Distância de cosseno: compara a direção dos vetores, não o tamanho.
        # É a métrica usual para embeddings de texto.
        self._collection = client.get_or_create_collection(
            name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )

    def count(self) -> int:
        return self._collection.count()

    def sync(self, chunks: list[Chunk]) -> SyncStats:
        """Deixa a coleção igual à lista de trechos, calculando embeddings só do que é novo.

        Funciona porque os IDs dos trechos são determinísticos (etapa 1): o mesmo trecho
        tem sempre o mesmo ID, então "já indexado" é só "ID já existe na coleção".
        """
        existing = set(self._collection.get(include=[])["ids"])
        wanted = {c.id: c for c in chunks}

        stale = sorted(existing - wanted.keys())
        if stale:
            self._collection.delete(ids=stale)

        new = [c for c in chunks if c.id not in existing]
        # Embeddings em blocos para não guardar milhares de vetores na memória de uma vez
        # e para o progresso ficar salvo se o processo for interrompido.
        block = 256
        for start in range(0, len(new), block):
            part = new[start : start + block]
            self._collection.add(
                ids=[c.id for c in part],
                embeddings=self.embedder.embed_documents([c.text for c in part]),
                documents=[c.text for c in part],
                metadatas=[
                    {"source": c.source, "location": c.location, "strategy": c.strategy}
                    for c in part
                ],
            )

        return SyncStats(added=len(new), removed=len(stale), kept=len(existing) - len(stale))

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        # O Chroma devolve listas de listas (uma por consulta); usamos só a primeira.
        rows = zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
        scored = [
            (
                Chunk(
                    id=chunk_id,
                    source=meta["source"],
                    location=meta["location"],
                    text=text,
                    strategy=meta["strategy"],
                ),
                1.0 - distance,  # distância de cosseno -> similaridade (1 = mesma direção)
            )
            for chunk_id, text, meta, distance in rows
        ]
        return to_results(scored)
