import json

from grounded_rag.evaluation.report import build_report


def summary(mean):
    return {"mean": mean, "ci95": [mean - 0.1, mean + 0.1], "n": 10}


def result(name, recall5, rerank, scores):
    questions = [{"answerable": answerable, "top_score": score} for answerable, score in scores]
    return {
        "experiment": {
            "name": name,
            "mode": "hybrid",
            "rerank": rerank,
            "embedding_model": "nomic-embed-text",
            "llm_model": None,
        },
        "index": "structure-200-40",
        "meta": {"n_questions": 3, "corpus_commit": "1a78ae3e4", "prompt_version": "1"},
        "aggregate": {
            "retrieval": {
                "recall@1": summary(0.5),
                "recall@5": summary(recall5),
                "recall@20": summary(0.9),
                "mrr@10": summary(0.6),
            },
            "retrieval_by_type": {k: None for k in ("exact", "paraphrase", "multi", "en", "pt")},
            "search_ms_p50": 12.0,
        },
        "questions": questions,
    }


def test_report_sorts_by_recall_and_formats_ci(tmp_path):
    scores = [(True, 5.0), (True, 3.0), (False, -2.0)]
    (tmp_path / "a.json").write_text(json.dumps(result("worse", 0.6, False, scores)))
    (tmp_path / "b.json").write_text(json.dumps(result("better", 0.8, True, scores)))

    report = build_report(tmp_path)

    assert report.index("| better |") < report.index("| worse |")
    assert "**80 (70–90)**" in report
    assert "## Calibração" in report
    # Limite 3.0 mantém as 2 respondíveis e barra a de fora do corpus.
    assert "| better | 3.00 | 100% | 100% |" in report
    assert "| worse | " not in report.split("## Calibração")[1]  # sem rerank, sem calibração


def test_empty_results_dir(tmp_path):
    assert "Nenhum resultado" in build_report(tmp_path)
