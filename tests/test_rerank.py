from grounded_rag.ingest.models import Chunk
from grounded_rag.retrieval.base import SearchResult, to_results
from grounded_rag.retrieval.rerank import RerankingRetriever


def chunk(id, text):
    return Chunk(id=id, source="doc.md", location=id, text=text, strategy="structure")


class FixedRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.requested_k = None

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.requested_k = k
        return to_results([(c, 1.0) for c in self.chunks])[:k]


CHUNKS = [
    chunk("generic", "security headers overview"),
    chunk("csrf", "csrf tokens protect state changing requests"),
    chunk("cookies", "samesite cookies help against csrf"),
]


def test_reranker_reorders_and_cuts_to_k(fake_reranker):
    base = FixedRetriever(CHUNKS)
    retriever = RerankingRetriever(base, fake_reranker, candidates=10)

    results = retriever.search("csrf tokens", k=2)

    # FakeReranker: nota = palavras da pergunta presentes no texto (csrf=2, cookies=1, generic=0)
    assert [r.chunk.id for r in results] == ["csrf", "cookies"]
    assert [r.score for r in results] == [2.0, 1.0]
    assert [r.rank for r in results] == [1, 2]
    assert base.requested_k == 10  # pede os candidatos, não só k


def test_rerank_keeps_origins_and_previous_rank(fake_reranker):
    retriever = RerankingRetriever(FixedRetriever(CHUNKS), fake_reranker, candidates=10)

    results = retriever.search("samesite cookies", k=1)

    assert results[0].chunk.id == "cookies"
    assert results[0].origins == {"before_rerank": 3}


def test_ties_keep_original_order(fake_reranker):
    retriever = RerankingRetriever(FixedRetriever(CHUNKS), fake_reranker, candidates=10)

    results = retriever.search("nothing matches", k=3)

    assert [r.chunk.id for r in results] == ["generic", "csrf", "cookies"]


def test_empty_candidates(fake_reranker):
    retriever = RerankingRetriever(FixedRetriever([]), fake_reranker, candidates=10)

    assert retriever.search("q", k=5) == []
