import pytest

from grounded_rag.ingest.chunking import chunk_segments, combine_locations
from grounded_rag.ingest.models import Segment


def seg(location, n_words, prefix="w", source="doc.md"):
    """Segmento com palavras numeradas (w0 w1 ...) para conferir exatamente os cortes."""
    return Segment(source, location, " ".join(f"{prefix}{i}" for i in range(n_words)))


def words(chunk):
    return chunk.text.split()


def test_fixed_windows_have_size_and_overlap():
    chunks = chunk_segments([seg("A", 10)], "fixed", size=4, overlap=1)

    # passo = 4 - 1 = 3 -> janelas [0,4) [3,7) [6,10)
    assert [words(c) for c in chunks] == [
        ["w0", "w1", "w2", "w3"],
        ["w3", "w4", "w5", "w6"],
        ["w6", "w7", "w8", "w9"],
    ]


def test_fixed_does_not_create_tiny_trailing_window():
    chunks = chunk_segments([seg("A", 5)], "fixed", size=4, overlap=1)

    # A segunda janela alcança o fim; não há uma terceira só com a sobreposição.
    assert [words(c) for c in chunks] == [["w0", "w1", "w2", "w3"], ["w3", "w4"]]


def test_fixed_crosses_segments_and_combines_locations():
    segments = [seg("A", 3, prefix="a"), seg("B", 3, prefix="b")]

    chunks = chunk_segments(segments, "fixed", size=4, overlap=0)

    assert [c.location for c in chunks] == ["A; B", "B"]
    assert words(chunks[0]) == ["a0", "a1", "a2", "b0"]


def test_structure_never_mixes_segments():
    segments = [seg("A", 3, prefix="a"), seg("B", 3, prefix="b")]

    chunks = chunk_segments(segments, "structure", size=4, overlap=0)

    assert [(c.location, words(c)) for c in chunks] == [
        ("A", ["a0", "a1", "a2"]),
        ("B", ["b0", "b1", "b2"]),
    ]


def test_structure_splits_large_segment_keeping_location():
    chunks = chunk_segments([seg("Big", 10)], "structure", size=4, overlap=1)

    assert len(chunks) == 3
    assert {c.location for c in chunks} == {"Big"}


def test_chunk_text_preserves_line_breaks():
    segment = Segment("doc.md", "A", "def f():\n    return 1")

    chunks = chunk_segments([segment], "structure", size=10, overlap=0)

    assert chunks[0].text == "def f():\n    return 1"


def test_ids_are_deterministic_and_unique():
    segments = [seg("A", 10), seg("B", 10)]

    first = chunk_segments(segments, "structure", size=4, overlap=1)
    second = chunk_segments(segments, "structure", size=4, overlap=1)

    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_same_text_in_other_strategy_gets_other_id():
    fixed = chunk_segments([seg("A", 3)], "fixed", size=4, overlap=0)
    structure = chunk_segments([seg("A", 3)], "structure", size=4, overlap=0)

    assert fixed[0].text == structure[0].text
    assert fixed[0].id != structure[0].id


@pytest.mark.parametrize("overlap", [-1, 4, 5])
def test_invalid_overlap_is_rejected(overlap):
    with pytest.raises(ValueError):
        chunk_segments([seg("A", 10)], "fixed", size=4, overlap=overlap)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        chunk_segments([seg("A", 10)], "semantic", size=4, overlap=0)


def test_page_locations_become_a_range():
    assert combine_locations(["p. 3", "p. 3", "p. 4"]) == "p. 3–4"


def test_section_locations_are_listed_once_each():
    assert combine_locations(["A", "A", "B"]) == "A; B"
