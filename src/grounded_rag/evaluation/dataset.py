"""Dataset de avaliação: perguntas rotuladas com as seções que contêm a resposta.

A relevância é rotulada por (fonte, seção) e não por ID de trecho: os IDs mudam com a
estratégia de chunking, a seção não. Assim o mesmo dataset avalia qualquer configuração.
"""

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, model_validator

from grounded_rag.ingest.readers import LOCATION_SEP, find_sections

QuestionType = Literal["exact", "paraphrase", "multi", "unanswerable"]


class RelevantSection(BaseModel):
    source: str  # ex.: "owasp/Password_Storage_Cheat_Sheet.md"
    section: str  # caminho de títulos; vale também para as subseções abaixo dele


class EvalQuestion(BaseModel):
    id: str
    question: str
    type: QuestionType
    lang: Literal["en", "pt"]
    answerable: bool
    relevant: list[RelevantSection] = []
    key_facts: list[str] = []  # trechos curtos que uma resposta correta deve conter
    origin: Literal["generated", "handwritten"]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.answerable and not self.relevant:
            raise ValueError(f"{self.id}: pergunta respondível precisa de seção relevante")
        if not self.answerable and (self.relevant or self.type != "unanswerable"):
            raise ValueError(f"{self.id}: pergunta fora do corpus não tem seção relevante")
        return self


def load_dataset(path: Path) -> list[EvalQuestion]:
    with path.open(encoding="utf-8") as f:
        questions = [EvalQuestion(**json.loads(line)) for line in f if line.strip()]
    ids = [q.id for q in questions]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"IDs repetidos no dataset: {duplicated}")
    return questions


def section_matches(location: str, section: str) -> bool:
    """A localização está dentro da seção? ("A › B › C" está dentro de "A › B")."""
    return location == section or location.startswith(section + LOCATION_SEP)


def unknown_sections(questions: list[EvalQuestion], data_dir: Path) -> list[str]:
    """Seções rotuladas que não existem no corpus (erro de digitação, corpus mudou...)."""
    sections_by_source: dict[str, list[str]] = {}
    problems = []
    for q in questions:
        for rel in q.relevant:
            if rel.source not in sections_by_source:
                path = data_dir / rel.source
                sections_by_source[rel.source] = find_sections(path) if path.exists() else []
            if not any(section_matches(s, rel.section) for s in sections_by_source[rel.source]):
                problems.append(f"{q.id}: {rel.source} › {rel.section}")
    return problems
