import pytest

from grounded_rag.cli import main
from grounded_rag.config import get_settings
from grounded_rag.ingest.pipeline import ingest_folder, read_jsonl, write_jsonl


@pytest.fixture
def corpus(tmp_path):
    data = tmp_path / "data"
    (data / "owasp").mkdir(parents=True)
    (data / "owasp" / "XSS.md").write_text(
        "# XSS\n\nVeja [o guia](https://x).\n\n## Encoding\n\nCodifique a saída.\n",
        encoding="utf-8",
    )
    (data / "owasp" / "MANIFEST.json").write_text("{}", encoding="utf-8")  # deve ser ignorado
    return data


def test_ingest_folder_reads_cleans_and_chunks(corpus):
    chunks = ingest_folder(corpus, "structure", size=50, overlap=0)

    assert [(c.source, c.location, c.text) for c in chunks] == [
        ("owasp/XSS.md", "XSS", "Veja o guia."),
        ("owasp/XSS.md", "XSS › Encoding", "Codifique a saída."),
    ]


def test_jsonl_round_trip(corpus, tmp_path):
    chunks = ingest_folder(corpus, "fixed", size=50, overlap=0)
    path = tmp_path / "out" / "chunks.jsonl"

    write_jsonl(chunks, path)

    assert read_jsonl(path) == chunks


@pytest.fixture
def cli_env(corpus, tmp_path, monkeypatch):
    # chdir para uma pasta temporária: o .env real do projeto não interfere no teste.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("DATA_DIR", str(corpus))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_cli_ingest_writes_chunks_and_prints_stats(cli_env, capsys):
    assert main(["ingest", "--strategy", "fixed", "--size", "50", "--overlap", "0"]) == 0

    out = capsys.readouterr().out
    assert "Documentos: 1  Trechos: 1" in out
    assert len(read_jsonl(cli_env / "processed" / "chunks.jsonl")) == 1


def test_cli_ingest_rejects_invalid_overlap(cli_env, capsys):
    assert main(["ingest", "--size", "10", "--overlap", "10"]) == 1
    assert "overlap" in capsys.readouterr().err
