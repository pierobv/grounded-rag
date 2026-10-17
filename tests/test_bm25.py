from grounded_rag.ingest.models import Chunk
from grounded_rag.retrieval.bm25 import BM25Index, tokenize


def chunk(id, text):
    return Chunk(id=id, source="doc.md", location=id, text=text, strategy="structure")


# Pelo menos 3 trechos: com só 2, um termo presente em 1 deles tem IDF = log(1) = 0
# na fórmula do BM25Okapi, e o score seria zero.
CHUNKS = [
    chunk("xss", "Set the Content-Security-Policy header to mitigate XSS."),
    chunk("clickjacking", "Use X-Frame-Options or frame-ancestors against clickjacking."),
    chunk("sqli", "Use prepared statements to prevent SQL injection."),
    chunk("passwords", "Store passwords with Argon2id."),
]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("X-Frame-Options: DENY") == ["x", "frame", "options", "deny"]


def test_tokenize_keeps_accented_words_whole():
    assert tokenize("Codifique a saída_html") == ["codifique", "a", "saída", "html"]


def test_exact_term_ranks_matching_chunk_first():
    results = BM25Index(CHUNKS).search("X-Frame-Options", k=3)

    assert results[0].chunk.id == "clickjacking"
    assert results[0].rank == 1


def test_chunks_without_query_terms_are_not_returned():
    results = BM25Index(CHUNKS).search("argon2id", k=10)

    assert [r.chunk.id for r in results] == ["passwords"]


def test_empty_query_or_index_returns_nothing():
    assert BM25Index(CHUNKS).search("!!!", k=5) == []
    assert BM25Index([]).search("xss", k=5) == []
