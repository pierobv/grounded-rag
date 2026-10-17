import pytest
from pydantic import ValidationError

from grounded_rag.generation.citations import DEFAULT_ABSTAIN_MESSAGE, Answer, build_answer
from grounded_rag.ingest.models import Chunk

CHUNKS = [
    Chunk(id=str(n), source=f"doc{n}.md", location=f"Sec {n}", text="...", strategy="structure")
    for n in range(1, 4)
]


def test_valid_citations_are_resolved_to_source_and_location():
    answer = build_answer("Use HttpOnly [2]. Rotate IDs [1][2].", CHUNKS)

    assert not answer.abstained
    assert [(c.number, c.source, c.location) for c in answer.citations] == [
        (2, "doc2.md", "Sec 2"),
        (1, "doc1.md", "Sec 1"),  # ordem da primeira aparição, sem repetir
    ]


def test_comma_separated_markers_are_supported():
    answer = build_answer("Both apply [1, 3].", CHUNKS)

    assert [c.number for c in answer.citations] == [1, 3]


def test_citation_to_missing_excerpt_is_removed_and_recorded():
    answer = build_answer("Fact [2]. Invented [7]. Mixed [3, 9].", CHUNKS)

    assert answer.text == "Fact [2]. Invented . Mixed [3]."
    assert answer.invalid_citations == [7, 9]
    assert [c.number for c in answer.citations] == [2, 3]


def test_answer_without_any_valid_citation_becomes_abstention():
    answer = build_answer("Paris is the capital of France.", CHUNKS)

    assert answer.abstained
    assert answer.text == DEFAULT_ABSTAIN_MESSAGE
    assert answer.raw_text == "Paris is the capital of France."  # guardado para depuração


def test_only_invalid_citations_also_abstains():
    answer = build_answer("Something [8].", CHUNKS)

    assert answer.abstained
    assert answer.citations == []


@pytest.mark.parametrize(
    "raw",
    [
        "INSUFFICIENT_CONTEXT The documents do not contain this information.",
        "INSUFFICIENT CONTEXT The documents do not contain this information.",  # llama3.2:3b
        "Insufficient-Context: The documents do not contain this information",
        "INSUFFICIENT_CONTEXT[1] The documents do not contain this information.",
    ],
)
def test_sentinel_variants_abstain_with_model_message(raw):
    answer = build_answer(raw, CHUNKS)

    assert answer.abstained
    assert answer.text == "The documents do not contain this information."


def test_bare_sentinel_uses_default_message():
    assert build_answer("INSUFFICIENT_CONTEXT", CHUNKS).text == DEFAULT_ABSTAIN_MESSAGE


def test_model_refuses_ungrounded_answer():
    # A regra também vale fora do parser: não dá para construir uma resposta assim.
    with pytest.raises(ValidationError, match="citação"):
        Answer(text="x", citations=[], abstained=False, raw_text="x")
