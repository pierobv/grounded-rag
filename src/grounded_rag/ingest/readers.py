"""Leitores: transformam um arquivo em uma lista de segmentos (página ou seção)."""

import re
from pathlib import Path

import pymupdf

from grounded_rag.ingest.models import Segment

# Título ATX: 1 a 6 "#", espaço, texto. "#hashtag" (sem espaço) não é título.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Abertura/fechamento de bloco de código: ``` ou ~~~ (com até 3 espaços de recuo).
FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")

LOCATION_SEP = " › "


def read_markdown(path: Path, source: str) -> list[Segment]:
    """Divide o Markdown em seções. A localização é o caminho de títulos até a seção.

    Mantemos uma pilha de títulos: ao encontrar um título de nível N, descartamos
    os títulos de nível >= N e empilhamos o novo. Ex.: "# A", "## B", "### C", "## D"
    gera as localizações "A", "A › B", "A › B › C", "A › D".
    """
    segments, _ = _parse_markdown(path, source)
    return segments


def find_sections(path: Path) -> list[str]:
    """Todas as localizações de títulos do arquivo, inclusive seções sem texto próprio.

    Usado para validar o dataset de avaliação: "A › B" pode não ter texto (só subseções),
    mas é um rótulo válido.
    """
    _, locations = _parse_markdown(path, source=path.name)
    return locations


def _parse_markdown(path: Path, source: str) -> tuple[list[Segment], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()

    segments: list[Segment] = []
    locations: list[str] = []
    headings: list[tuple[int, str]] = []  # pilha de (nível, título)
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            location = LOCATION_SEP.join(title for _, title in headings) or path.stem
            segments.append(Segment(source=source, location=location, text=text))
        buffer.clear()

    for line in lines:
        if FENCE_RE.match(line):
            in_code = not in_code
            buffer.append(line)
            continue

        # Dentro de bloco de código, "# comentário" é código, não título.
        match = None if in_code else HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
            locations.append(LOCATION_SEP.join(t for _, t in headings))
        else:
            buffer.append(line)

    flush()
    return segments, locations


def read_pdf(path: Path, source: str) -> list[Segment]:
    """Um segmento por página. Páginas numeradas a partir de 1, como o leitor humano vê."""
    segments = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():  # páginas em branco ou só com imagem não viram segmento
                segments.append(Segment(source=source, location=f"p. {number}", text=text))
    return segments


READERS = {".md": read_markdown, ".pdf": read_pdf}


def read_document(path: Path, root: Path) -> list[Segment]:
    """Escolhe o leitor pela extensão. `source` é o caminho relativo à pasta de dados."""
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Formato não suportado: {path.suffix}")
    return reader(path, source=path.relative_to(root).as_posix())
