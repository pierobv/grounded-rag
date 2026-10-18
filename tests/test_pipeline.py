import pytest

from grounded_rag.cli import main
from grounded_rag.config import get_settings
from grounded_rag.ingest.pipeline import ingest_folder, read_jsonl, write_jsonl
from grounded_rag.retrieval import factory


@pytest.fixture
def corpus(tmp_path):
    data = tmp_path / "data"
    (data / "owasp").mkdir(parents=True)
    (data / "owasp" / "XSS.md").write_text(
        "# XSS\n\nVeja [o guia](https://x).\n\n## Encoding\n\nCodifique a saída.\n",
        encoding="utf-8",
    )
    # Um segundo documento: com poucos trechos, o IDF do BM25Okapi pode zerar.
    (data / "owasp" / "SQLi.md").write_text(
        "# SQL Injection\n\nUse consultas parametrizadas.\n", encoding="utf-8"
    )
    (data / "owasp" / "MANIFEST.json").write_text("{}", encoding="utf-8")  # deve ser ignorado
    return data


def test_ingest_folder_reads_cleans_and_chunks(corpus):
    chunks = ingest_folder(corpus, "structure", size=50, overlap=0)

    assert [(c.source, c.location, c.text) for c in chunks] == [
        ("owasp/SQLi.md", "SQL Injection", "Use consultas parametrizadas."),
        ("owasp/XSS.md", "XSS", "Veja o guia."),
        ("owasp/XSS.md", "XSS › Encoding", "Codifique a saída."),
    ]


def test_jsonl_round_trip(corpus, tmp_path):
    chunks = ingest_folder(corpus, "fixed", size=50, overlap=0)
    path = tmp_path / "out" / "chunks.jsonl"

    write_jsonl(chunks, path)

    assert read_jsonl(path) == chunks


@pytest.fixture
def cli_env(corpus, tmp_path, monkeypatch, fake_embedder, fake_reranker, fake_llm):
    # chdir para uma pasta temporária: o .env real do projeto não interfere no teste.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434")
    monkeypatch.setenv("DATA_DIR", str(corpus))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    # A fábrica cria embedder e reranker por estas funções; trocamos pelos falsos
    # (sem Ollama e sem baixar modelo do Hugging Face).
    monkeypatch.setattr(factory, "make_embedder", lambda settings: fake_embedder)
    monkeypatch.setattr(factory, "make_reranker", lambda settings: fake_reranker)
    monkeypatch.setattr(factory, "make_llm", lambda settings, model: fake_llm)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


INGEST_FIXED = ["ingest", "--strategy", "fixed", "--size", "50", "--overlap", "0"]


def test_cli_ingest_writes_chunks_indexes_and_prints_stats(cli_env, capsys):
    assert main(INGEST_FIXED) == 0

    out = capsys.readouterr().out
    assert "Documentos: 2  Trechos: 2" in out
    assert "Índice vetorial: 2 novos, 0 removidos, 0 mantidos" in out
    assert len(read_jsonl(cli_env / "processed" / "chunks-fixed-50-0-h.jsonl")) == 2


def test_cli_ingest_no_index_skips_embeddings(cli_env, capsys, fake_embedder):
    assert main([*INGEST_FIXED, "--no-index"]) == 0

    assert fake_embedder.embedded_texts == 0
    assert "Índice vetorial" not in capsys.readouterr().out


SIZE = ["--size", "50", "--overlap", "0"]


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
@pytest.mark.parametrize("rerank", ["--rerank", "--no-rerank"])
def test_cli_search_after_ingest(cli_env, capsys, mode, rerank):
    main(["ingest", *SIZE])
    capsys.readouterr()  # descarta a saída do ingest

    code = main(["search", "codifique a saída", *SIZE, "--mode", mode, rerank])

    out = capsys.readouterr().out
    assert code == 0
    assert "1. [" in out
    assert "owasp/XSS.md — XSS › Encoding" in out.split("2. [")[0]  # o certo vem em 1º


def test_cli_hybrid_rerank_shows_origins(cli_env, capsys):
    main(["ingest", *SIZE])
    capsys.readouterr()

    main(["search", "codifique a saída", *SIZE, "--mode", "hybrid", "--rerank"])

    out = capsys.readouterr().out
    assert "Busca hybrid + rerank" in out
    assert "vetorial #" in out and "bm25 #" in out and "antes do rerank #" in out


def test_cli_search_uses_settings_defaults(cli_env, capsys, fake_reranker):
    main(["ingest", *SIZE])
    capsys.readouterr()

    main(["search", "codifique a saída", *SIZE])  # padrão: hybrid + rerank

    assert "Busca hybrid + rerank" in capsys.readouterr().out
    assert len(fake_reranker.calls) == 1


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_cli_search_without_ingest_explains_what_to_do(cli_env, capsys, mode):
    assert main(["search", "xss", "--mode", mode]) == 1
    assert "grounded-rag ingest" in capsys.readouterr().err


def test_cli_ingest_rejects_invalid_overlap(cli_env, capsys):
    assert main(["ingest", "--size", "10", "--overlap", "10"]) == 1
    assert "overlap" in capsys.readouterr().err


def test_cli_ask_streams_answer_and_lists_sources(cli_env, capsys, fake_llm):
    main(["ingest", *SIZE])
    capsys.readouterr()

    assert main(["ask", "codifique a saída", *SIZE, "--show-context"]) == 0

    out = capsys.readouterr().out
    assert "Trechos enviados ao modelo:" in out
    assert "Use HttpOnly cookies [1]." in out
    assert "Fontes:\n  [1] owasp/" in out
    assert "fake-llm" in out


def test_cli_ask_discards_uncited_answer(cli_env, capsys, fake_llm):
    main(["ingest", *SIZE])
    capsys.readouterr()
    fake_llm.reply = "Paris is the capital of France."

    main(["ask", "capital of France?", *SIZE])

    out = capsys.readouterr().out
    assert "Resposta descartada" in out
    assert "Não encontrei essa informação nos documentos." in out


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], "qwen2.5:7b"), (["--dev"], "llama3.2:3b"), (["--model", "x:1b"], "x:1b")],
)
def test_cli_ask_model_selection(cli_env, capsys, monkeypatch, fake_llm, flags, expected):
    chosen = []
    monkeypatch.setattr(
        factory, "make_llm", lambda settings, model: chosen.append(model) or fake_llm
    )
    main(["ingest", *SIZE])

    main(["ask", "q", *SIZE, *flags])

    assert chosen == [expected]
