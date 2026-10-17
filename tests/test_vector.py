import pytest

from grounded_rag.ingest.models import Chunk
from grounded_rag.retrieval.vector import VectorIndex, collection_name


def chunk(id, text, location="A"):
    return Chunk(id=id, source="doc.md", location=location, text=text, strategy="structure")


CHUNKS = [
    chunk("1", "cross site scripting output encoding html", "XSS"),
    chunk("2", "sql injection parameterized queries prepared statements", "SQLi"),
    chunk("3", "password storage hashing argon2 bcrypt", "Passwords"),
]


@pytest.fixture
def index(tmp_path, fake_embedder):
    return VectorIndex(tmp_path / "chroma", "test-index", fake_embedder)


def test_nearest_chunk_comes_first_with_metadata(index):
    index.sync(CHUNKS)

    results = index.search("prevent sql injection with prepared statements", k=2)

    assert results[0].chunk == CHUNKS[1]  # texto e metadados voltam intactos do Chroma
    assert [r.rank for r in results] == [1, 2]
    assert results[0].score > results[1].score


def test_resync_does_not_embed_again(index, fake_embedder):
    index.sync(CHUNKS)
    stats = index.sync(CHUNKS)

    assert (stats.added, stats.removed, stats.kept) == (0, 0, 3)
    assert fake_embedder.embedded_texts == 3  # só a primeira sincronização calculou


def test_sync_removes_stale_and_adds_new(index, fake_embedder):
    index.sync(CHUNKS)
    new = chunk("4", "csrf tokens samesite cookies", "CSRF")

    stats = index.sync([*CHUNKS[:2], new])

    assert (stats.added, stats.removed, stats.kept) == (1, 1, 2)
    assert index.count() == 3
    assert fake_embedder.embedded_texts == 4


def test_index_persists_on_disk(tmp_path, fake_embedder):
    VectorIndex(tmp_path / "chroma", "test-index", fake_embedder).sync(CHUNKS)

    reopened = VectorIndex(tmp_path / "chroma", "test-index", fake_embedder)

    assert reopened.count() == 3


def test_empty_index_returns_no_results(index):
    assert index.search("anything", k=5) == []


def test_collection_name_is_valid_for_chroma():
    assert collection_name("structure-200-40", "nomic-embed-text:latest") == (
        "structure-200-40-nomic-embed-text"
    )
    assert (
        collection_name("fixed-100-0", "qwen3-embedding:0.6b") == "fixed-100-0-qwen3-embedding-0.6b"
    )
