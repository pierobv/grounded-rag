"""Pipeline de ingestão: ler -> limpar -> dividir em trechos -> salvar em JSONL."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from grounded_rag.ingest.chunking import chunk_segments
from grounded_rag.ingest.clean import CLEANERS
from grounded_rag.ingest.models import Chunk
from grounded_rag.ingest.readers import READERS, read_document


@dataclass(frozen=True)
class IngestStats:
    documents: int
    chunks: int
    avg_words: float
    min_words: int
    max_words: int


def find_documents(data_dir: Path) -> list[Path]:
    """Arquivos suportados, em ordem estável (a ordem afeta os IDs e a reprodutibilidade)."""
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in READERS)


def ingest_folder(data_dir: Path, strategy: str, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in find_documents(data_dir):
        clean = CLEANERS[path.suffix.lower()]
        segments = [replace(s, text=clean(s.text)) for s in read_document(path, root=data_dir)]
        segments = [s for s in segments if s.text]  # a limpeza pode esvaziar um segmento
        chunks.extend(chunk_segments(segments, strategy, size, overlap))
    return chunks


def write_jsonl(chunks: list[Chunk], path: Path) -> None:
    """JSONL: um objeto JSON por linha. Fácil de inspecionar, de ler em streaming e de versionar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            # ensure_ascii=False mantém acentos legíveis no arquivo.
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def compute_stats(chunks: list[Chunk]) -> IngestStats:
    sizes = [c.n_words for c in chunks] or [0]
    return IngestStats(
        documents=len({c.source for c in chunks}),
        chunks=len(chunks),
        avg_words=sum(sizes) / len(sizes),
        min_words=min(sizes),
        max_words=max(sizes),
    )
