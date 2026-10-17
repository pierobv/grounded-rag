"""Chunking: transforma os segmentos de um documento em trechos para indexar.

Duas estratégias, comparadas depois nas avaliações:

- fixed:     janelas de N palavras com sobreposição, ignorando a estrutura do documento.
             É a linha de base. Um trecho pode atravessar páginas ou seções.
- structure: um trecho por segmento (seção ou página); segmentos grandes demais são
             divididos em janelas, sem nunca misturar dois segmentos.

O tamanho é medido em palavras (sequências sem espaço), não em tokens de um modelo.
Assim o chunking não depende do modelo de embedding escolhido.
"""

import hashlib
import re
from dataclasses import dataclass

from grounded_rag.ingest.models import Chunk, Segment

WORD_RE = re.compile(r"\S+")
SEGMENT_SEP = "\n\n"
PAGE_PREFIX = "p. "


@dataclass(frozen=True)
class _Word:
    start: int  # posição do primeiro caractere no texto concatenado
    end: int  # posição logo após o último caractere
    segment: int  # índice do segmento de onde a palavra veio


def _windows(n_words: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Intervalos [início, fim) de palavras de cada janela.

    Ex.: n_words=10, size=4, overlap=1 -> passo 3 -> (0,4) (3,7) (6,10).
    Paramos assim que uma janela alcança o fim, para não gerar uma última janela
    minúscula feita só da sobreposição.
    """
    step = size - overlap
    windows = []
    for start in range(0, n_words, step):
        end = min(start + size, n_words)
        windows.append((start, end))
        if end == n_words:
            break
    return windows


def combine_locations(locations: list[str]) -> str:
    """Localização de um trecho que atravessa segmentos.

    Páginas viram intervalo ("p. 3–4"); seções são listadas ("A › B; A › C").
    """
    unique = list(dict.fromkeys(locations))  # remove repetidos mantendo a ordem
    if len(unique) == 1:
        return unique[0]
    if all(loc.startswith(PAGE_PREFIX) for loc in unique):
        return f"{unique[0]}–{unique[-1].removeprefix(PAGE_PREFIX)}"
    return "; ".join(unique)


def _chunk_id(source: str, strategy: str, index: int, text: str) -> str:
    """ID determinístico: o mesmo documento e configuração geram sempre os mesmos IDs.

    Isso permite reindexar sem duplicar trechos no banco vetorial (etapa 2).
    """
    key = f"{source}|{strategy}|{index}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _sliding_chunks(segments: list[Segment], size: int, overlap: int, strategy: str) -> list[Chunk]:
    """Janelas deslizantes sobre o texto concatenado dos segmentos.

    Guardamos, para cada palavra, sua posição em caracteres e o segmento de origem.
    O texto do trecho é um recorte do texto original (preserva quebras de linha e
    recuo de código), e a localização sai dos segmentos que as palavras tocam.
    """
    text_parts: list[str] = []
    words: list[_Word] = []
    offset = 0
    for index, segment in enumerate(segments):
        for match in WORD_RE.finditer(segment.text):
            words.append(_Word(offset + match.start(), offset + match.end(), index))
        text_parts.append(segment.text)
        offset += len(segment.text) + len(SEGMENT_SEP)
    full_text = SEGMENT_SEP.join(text_parts)

    if not words:
        return []

    source = segments[0].source
    chunks = []
    for start, end in _windows(len(words), size, overlap):
        window = words[start:end]
        text = full_text[window[0].start : window[-1].end]
        location = combine_locations([segments[w.segment].location for w in window])
        chunks.append(
            Chunk(
                id=_chunk_id(source, strategy, len(chunks), text),
                source=source,
                location=location,
                text=text,
                strategy=strategy,
            )
        )
    return chunks


def fixed_chunks(segments: list[Segment], size: int, overlap: int) -> list[Chunk]:
    return _sliding_chunks(segments, size, overlap, strategy="fixed")


def structure_chunks(segments: list[Segment], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for segment in segments:
        # Cada segmento é processado sozinho: um trecho nunca mistura duas seções/páginas.
        # Segmento pequeno -> uma janela só; grande -> várias janelas com sobreposição.
        for piece in _sliding_chunks([segment], size, overlap, strategy="structure"):
            # Renumera os IDs no documento todo (cada chamada recomeça do índice 0).
            chunks.append(
                Chunk(
                    id=_chunk_id(piece.source, piece.strategy, len(chunks), piece.text),
                    source=piece.source,
                    location=piece.location,
                    text=piece.text,
                    strategy=piece.strategy,
                )
            )
    return chunks


STRATEGIES = {"fixed": fixed_chunks, "structure": structure_chunks}


def chunk_segments(segments: list[Segment], strategy: str, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError(f"overlap deve estar entre 0 e size-1 (size={size}, overlap={overlap})")
    if strategy not in STRATEGIES:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    return STRATEGIES[strategy](segments, size, overlap)
