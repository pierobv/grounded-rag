import json

import pytest
from pydantic import ValidationError

from grounded_rag.evaluation.dataset import EvalQuestion, load_dataset, unknown_sections


def question(**kwargs):
    base = {
        "id": "q1",
        "question": "How?",
        "type": "exact",
        "lang": "en",
        "answerable": True,
        "relevant": [{"source": "owasp/XSS.md", "section": "XSS › Encoding"}],
        "origin": "handwritten",
    }
    return base | kwargs


def test_answerable_question_needs_relevant_section():
    with pytest.raises(ValidationError, match="seção relevante"):
        EvalQuestion(**question(relevant=[]))


def test_unanswerable_question_has_no_relevant_section():
    with pytest.raises(ValidationError):
        EvalQuestion(**question(answerable=False, type="unanswerable"))

    EvalQuestion(**question(answerable=False, type="unanswerable", relevant=[]))  # válido


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(question()) + "\n" + json.dumps(question()) + "\n")

    with pytest.raises(ValueError, match="q1"):
        load_dataset(path)


def test_unknown_sections_finds_typos(tmp_path):
    (tmp_path / "owasp").mkdir()
    (tmp_path / "owasp" / "XSS.md").write_text("# XSS\n\n## Encoding\n\ntext\n")
    ok = EvalQuestion(**question())
    typo = EvalQuestion(
        **question(id="q2", relevant=[{"source": "owasp/XSS.md", "section": "XSS › Encodnig"}])
    )
    parent = EvalQuestion(
        **question(id="q3", relevant=[{"source": "owasp/XSS.md", "section": "XSS"}])
    )

    assert unknown_sections([ok, typo, parent], tmp_path) == ["q2: owasp/XSS.md › XSS › Encodnig"]
