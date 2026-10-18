"""Fidelidade das citações com um LLM como juiz.

A validação da etapa 4 garante que a citação EXISTE. Aqui medimos se o trecho citado
SUSTENTA a afirmação: para cada frase com citação, o juiz recebe a frase e os trechos
citados e responde se a frase é sustentada por eles.

Viés conhecido: o juiz é um LLM (e pode ser o mesmo que gerou a resposta). Por isso a
concordância do juiz com rótulos humanos é medida à parte (evals/judge_labels.jsonl).
"""

import re
from dataclasses import dataclass
from typing import Protocol

from grounded_rag.generation.citations import MARKER_RE
from grounded_rag.generation.llm import OllamaChat
from grounded_rag.ingest.models import Chunk

# Fim de frase seguido de espaço, ou quebra de linha (itens de lista viram frases próprias).
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# Marcadores no fim de uma frase ficam com ela: "X [1]. Y [2]." -> "X [1]." e "Y [2]."
MIN_CLAIM_WORDS = 3

JUDGE_PROMPT = """You check whether a claim is supported by source excerpts.
Answer "supported": true only if the excerpts explicitly state or directly imply the claim.
If the claim adds facts that are not in the excerpts, answer false.
First write a one-sentence reason, then the verdict."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},  # o motivo vem antes: o modelo "pensa" antes do veredito
        "supported": {"type": "boolean"},
    },
    "required": ["reason", "supported"],
}


@dataclass(frozen=True)
class Claim:
    text: str  # frase sem os marcadores
    cited: list[int]  # números dos trechos citados


@dataclass(frozen=True)
class Verdict:
    claim: str
    cited: list[int]
    supported: bool
    reason: str
    excerpts: list[str]  # trechos que o juiz viu (truncados): permite auditar o veredito


EXCERPT_AUDIT_CHARS = 800


class Judge(Protocol):
    def judge(self, claim: str, excerpts: list[str]) -> tuple[bool, str]: ...


class OllamaJudge:
    def __init__(self, llm: OllamaChat) -> None:
        self.llm = llm

    def judge(self, claim: str, excerpts: list[str]) -> tuple[bool, str]:
        sources = "\n\n".join(f"Excerpt:\n{e}" for e in excerpts)
        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"{sources}\n\nClaim: {claim}"},
        ]
        result = self.llm.complete_json(messages, JUDGE_SCHEMA)
        return bool(result["supported"]), str(result["reason"])


def split_claims(answer_text: str) -> tuple[list[Claim], int]:
    """Divide a resposta em frases. Devolve (frases com citação, nº de frases sem citação).

    Frases muito curtas ("Example:", "1.") são ignoradas: não são afirmações.
    """
    claims: list[Claim] = []
    uncited = 0
    for sentence in SENTENCE_SPLIT_RE.split(answer_text):
        numbers = [int(n) for m in MARKER_RE.finditer(sentence) for n in m.group(1).split(",")]
        text = " ".join(MARKER_RE.sub("", sentence).split())
        if len(text.split()) < MIN_CLAIM_WORDS:
            continue
        if numbers:
            claims.append(Claim(text, list(dict.fromkeys(numbers))))
        else:
            uncited += 1
    return claims, uncited


def judge_answer(answer_text: str, chunks: list[Chunk], judge: Judge) -> tuple[list[Verdict], int]:
    """Julga cada frase citada contra os trechos que ela cita."""
    claims, uncited = split_claims(answer_text)
    verdicts = []
    for claim in claims:
        excerpts = [chunks[n - 1].text for n in claim.cited if 1 <= n <= len(chunks)]
        if not excerpts:
            continue
        supported, reason = judge.judge(claim.text, excerpts)
        audit = [e[:EXCERPT_AUDIT_CHARS] for e in excerpts]
        verdicts.append(Verdict(claim.text, claim.cited, supported, reason, audit))
    return verdicts, uncited
