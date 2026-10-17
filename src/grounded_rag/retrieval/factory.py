"""Monta o buscador a partir da configuração. Usado pelo CLI e, depois, pela API e pela avaliação.

vector | bm25 | hybrid (vector + bm25 com RRF)   [+ reranking com cross-encoder]
"""

from dataclasses import dataclass
from pathlib import Path

from grounded_rag.config import Settings
from grounded_rag.generation.llm import LLM, OllamaChat
from grounded_rag.ingest.pipeline import read_jsonl
from grounded_rag.retrieval.base import Retriever
from grounded_rag.retrieval.bm25 import BM25Index
from grounded_rag.retrieval.embeddings import Embedder, OllamaEmbedder
from grounded_rag.retrieval.hybrid import HybridRetriever
from grounded_rag.retrieval.rerank import CrossEncoderReranker, Reranker, RerankingRetriever
from grounded_rag.retrieval.vector import VectorIndex, collection_name


class IndexNotReadyError(RuntimeError):
    """O índice pedido ainda não existe: é preciso rodar `grounded-rag ingest`."""


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    size: int
    overlap: int

    @property
    def index_name(self) -> str:
        return f"{self.strategy}-{self.size}-{self.overlap}"


def chunks_path(settings: Settings, config: ChunkingConfig) -> Path:
    return settings.processed_dir / f"chunks-{config.index_name}.jsonl"


def make_embedder(settings: Settings) -> Embedder:
    """Ponto único de criação do embedder (os testes o substituem por um falso)."""
    return OllamaEmbedder(
        base_url=str(settings.ollama_base_url),
        model=settings.embedding_model,
        doc_prefix=settings.embed_doc_prefix,
        query_prefix=settings.embed_query_prefix,
        batch_size=settings.embed_batch_size,
    )


def make_llm(settings: Settings, model: str) -> LLM:
    """Ponto único de criação do LLM (os testes o substituem por um falso)."""
    return OllamaChat(
        base_url=str(settings.ollama_base_url),
        model=model,
        num_ctx=settings.num_ctx,
        temperature=settings.temperature,
        seed=settings.seed,
    )


def make_reranker(settings: Settings) -> Reranker:
    """Ponto único de criação do reranker (os testes o substituem por um falso)."""
    return CrossEncoderReranker(settings.rerank_model)


def open_vector_index(settings: Settings, config: ChunkingConfig) -> VectorIndex:
    name = collection_name(config.index_name, settings.embedding_model)
    return VectorIndex(settings.chroma_dir, name, make_embedder(settings))


def _vector(settings: Settings, config: ChunkingConfig) -> VectorIndex:
    index = open_vector_index(settings, config)
    if index.count() == 0:
        raise IndexNotReadyError("Índice vetorial vazio. Rode `grounded-rag ingest`.")
    return index


def _bm25(settings: Settings, config: ChunkingConfig) -> BM25Index:
    path = chunks_path(settings, config)
    if not path.exists():
        raise IndexNotReadyError(f"Trechos não encontrados ({path}). Rode `grounded-rag ingest`.")
    return BM25Index(read_jsonl(path))


def build_retriever(
    settings: Settings, config: ChunkingConfig, mode: str, rerank: bool
) -> Retriever:
    retriever: Retriever
    if mode == "vector":
        retriever = _vector(settings, config)
    elif mode == "bm25":
        retriever = _bm25(settings, config)
    elif mode == "hybrid":
        retriever = HybridRetriever(
            {"vector": _vector(settings, config), "bm25": _bm25(settings, config)},
            candidates=settings.candidates,
            k=settings.rrf_k,
        )
    else:
        raise ValueError(f"Modo de busca desconhecido: {mode}")

    if rerank:
        retriever = RerankingRetriever(retriever, make_reranker(settings), settings.candidates)
    return retriever
