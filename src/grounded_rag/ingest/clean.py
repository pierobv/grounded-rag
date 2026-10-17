"""Limpeza de texto antes do chunking.

Regra geral: remover o que é ruído para a busca (URLs, marcação de imagem, comentários),
sem perder conteúdo. Blocos de código são mantidos: nos cheat sheets, o código
muitas vezes é a própria resposta.
"""

import re
import unicodedata

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
# Espaços repetidos NO MEIO da linha. O recuo inicial é preservado (código Python depende dele).
SPACES_RE = re.compile(r"(?<=\S)[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Comum a todos os formatos.

    NFKC unifica caracteres equivalentes: ligaduras de PDF ("ﬁ" -> "fi"),
    espaço não separável -> espaço comum, etc. Sem isso, "ﬁle" e "file" seriam
    palavras diferentes para o BM25.
    """
    text = unicodedata.normalize("NFKC", text)
    lines = [SPACES_RE.sub(" ", line).rstrip() for line in text.splitlines()]
    return BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def clean_markdown(text: str) -> str:
    text = HTML_COMMENT_RE.sub("", text)
    text = MD_IMAGE_RE.sub("", text)  # antes dos links: imagem é "![...](...)"
    text = HTML_IMG_RE.sub("", text)
    text = MD_LINK_RE.sub(r"\1", text)  # [texto](url) -> texto
    return normalize(text)


def clean_pdf(text: str) -> str:
    # "vali-\ndation" -> "validation": palavra quebrada no fim da linha do PDF.
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    return normalize(text)


CLEANERS = {".md": clean_markdown, ".pdf": clean_pdf}
