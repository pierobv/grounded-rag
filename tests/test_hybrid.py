import pytest

from grounded_rag.ingest.models import Chunk
from grounded_rag.retrieval.base import SearchResult, to_results
from grounded_rag.retrieval.hybrid import HybridRetriever, rrf


def chunk(id):
    return Chunk(id=id, source="doc.md", location=id, text=f"text {id}", strategy="structure")


def ranking(*ids):
    """Lista de resultados na ordem dada (score irrelevante para o RRF)."""
    return to_results([(chunk(i), 0.0) for i in ids])


def test_rrf_scores_match_the_formula():
    fused = rrf({"vector": ranking("a", "b"), "bm25": ranking("b", "c")}, k=60)

    scores = {r.chunk.id: r.score for r in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)  # 2º no vetorial + 1º no BM25
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert [r.chunk.id for r in fused] == ["b", "a", "c"]
    assert [r.rank for r in fused] == [1, 2, 3]


def test_appearing_in_both_lists_beats_top_of_one():
    # "x" é 5º nas duas listas; "a" e "v" são 1º em uma só.
    fused = rrf(
        {"vector": ranking("v", "1", "2", "3", "x"), "bm25": ranking("a", "4", "5", "6", "x")},
        k=60,
    )

    assert fused[0].chunk.id == "x"  # 2/65 ≈ 0,0308 > 1/61 ≈ 0,0164


def test_origins_record_rank_in_each_list():
    fused = rrf({"vector": ranking("a", "b"), "bm25": ranking("b")}, k=60)

    origins = {r.chunk.id: r.origins for r in fused}
    assert origins == {"b": {"vector": 2, "bm25": 1}, "a": {"vector": 1}}


def test_ties_are_broken_deterministically_by_id():
    fused = rrf({"vector": ranking("z"), "bm25": ranking("a")}, k=60)

    assert [r.chunk.id for r in fused] == ["a", "z"]


def test_empty_rankings_give_empty_result():
    assert rrf({"vector": [], "bm25": []}) == []


class FixedRetriever:
    def __init__(self, ids):
        self.ids = ids
        self.requested_k = None

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.requested_k = k
        return ranking(*self.ids)[:k]


def test_hybrid_asks_each_retriever_for_candidates_and_cuts_to_k():
    vector, bm25 = FixedRetriever(["a", "b", "c"]), FixedRetriever(["c", "d"])
    hybrid = HybridRetriever({"vector": vector, "bm25": bm25}, candidates=20)

    results = hybrid.search("q", k=2)

    assert vector.requested_k == bm25.requested_k == 20
    assert [r.chunk.id for r in results] == ["c", "a"]
