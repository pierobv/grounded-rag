"""Tipos de dados da ingestão.

Dataclasses congeladas (frozen): leves, imutáveis e sem validação em tempo de execução.
Aqui os dados são criados pelo nosso próprio código, então não há entrada externa a validar;
o Pydantic fica nas fronteiras (API, citações geradas pelo LLM).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """Unidade natural de um documento: uma página (PDF) ou uma seção (Markdown)."""

    source: str  # caminho do arquivo relativo à pasta de dados, ex.: "owasp/XSS.md"
    location: str  # onde está no documento, ex.: "p. 3" ou "XSS › Output Encoding"
    text: str


@dataclass(frozen=True)
class Chunk:
    """Trecho que será indexado e recuperado. Carrega o necessário para a citação."""

    id: str
    source: str
    location: str
    text: str
    strategy: str  # "fixed" ou "structure"

    @property
    def n_words(self) -> int:
        return len(self.text.split())
