import json

import pytest

from grounded_rag.cli import main
from grounded_rag.config import get_settings
from grounded_rag.evaluation.runner import load_experiments
from grounded_rag.retrieval import factory

EXPERIMENTS = """
[defaults]
size = 50
overlap = 0

[[experiment]]
name = "bm25"
mode = "bm25"
rerank = false

[[experiment]]
name = "hybrid-llm"
llm_model = "fake"
judge = true
"""


@pytest.fixture
def eval_env(tmp_path, monkeypatch, fake_embedder, fake_reranker, fake_llm):
    data = tmp_path / "data" / "owasp"
    data.mkdir(parents=True)
    (data / "XSS.md").write_text("# XSS\n\n## Encoding\n\nEncode output in HTML context.\n")
    (data / "SQLi.md").write_text("# SQLi\n\n## Queries\n\nUse parameterized queries always.\n")
    (data / "CSRF.md").write_text("# CSRF\n\n## Tokens\n\nUse synchronizer tokens.\n")
    rows = [
        {
            "id": "a1",
            "question": "how to encode output html",
            "type": "exact",
            "lang": "en",
            "answerable": True,
            "relevant": [{"source": "owasp/XSS.md", "section": "XSS › Encoding"}],
            "key_facts": ["HttpOnly"],
            "origin": "handwritten",
        },
        {
            "id": "u1",
            "question": "capital of France",
            "type": "unanswerable",
            "lang": "en",
            "answerable": False,
            "origin": "handwritten",
        },
    ]
    (tmp_path / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "experiments.toml").write_text(EXPERIMENTS)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(factory, "make_embedder", lambda settings: fake_embedder)
    monkeypatch.setattr(factory, "make_reranker", lambda settings: fake_reranker)
    monkeypatch.setattr(factory, "make_llm", lambda settings, model: fake_llm)

    class AlwaysSupported:
        def judge(self, claim, excerpts):
            return True, "ok"

    monkeypatch.setattr(factory, "make_judge", lambda settings: AlwaysSupported())
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


ARGS = ["eval", "--file", "experiments.toml", "--dataset", "dataset.jsonl", "--results", "out"]


def test_load_experiments_applies_defaults(eval_env):
    experiments = load_experiments(eval_env / "experiments.toml")

    assert [(e.name, e.size, e.mode) for e in experiments] == [
        ("bm25", 50, "bm25"),
        ("hybrid-llm", 50, "hybrid"),
    ]


def test_eval_runs_experiments_saves_results_and_report(eval_env, capsys):
    assert main(ARGS) == 0

    bm25 = json.loads((eval_env / "out" / "bm25.json").read_text())
    assert bm25["aggregate"]["retrieval"]["recall@1"]["mean"] == 1.0
    assert bm25["meta"]["n_questions"] == 2
    assert "abstained" not in bm25["questions"][0]  # sem LLM

    hybrid = json.loads((eval_env / "out" / "hybrid-llm.json").read_text())
    answered = hybrid["questions"][0]
    assert answered["abstained"] is False  # FakeLLM responde "Use HttpOnly cookies [1]."
    assert answered["key_fact_recall"] == 1.0
    assert answered["faithfulness"] == 1.0
    gen = hybrid["aggregate"]["generation"]
    assert gen["answer_rate_answerable"]["mean"] == 1.0
    assert gen["refusal_rate_unanswerable"]["mean"] == 0.0  # o falso também "responde" a França

    report = (eval_env / "out" / "README.md").read_text()
    assert "| bm25 |" in report and "## Resposta" in report


def test_eval_retrieval_only_skips_llm(eval_env, fake_llm):
    assert main([*ARGS, "--retrieval-only", "--experiments", "hybrid-llm"]) == 0

    assert fake_llm.calls == []


def test_eval_rejects_unknown_experiment(eval_env, capsys):
    assert main([*ARGS, "--experiments", "nope"]) == 1
    assert "nope" in capsys.readouterr().err


def test_eval_rejects_dataset_with_unknown_section(eval_env, capsys):
    path = eval_env / "dataset.jsonl"
    # json.dumps grava "›" como \\u203a; trocamos só o fim do rótulo da seção.
    path.write_text(path.read_text().replace('Encoding"}', 'Encodnig"}'))

    assert main(ARGS) == 1
    assert "Encodnig" in capsys.readouterr().err
