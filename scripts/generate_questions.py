"""Gera perguntas CANDIDATAS para o dataset de avaliação (revisão humana obrigatória depois).

Uso:  uv run python scripts/generate_questions.py [--n 80] [--model qwen2.5:7b]
Saída: evals/candidates.jsonl (não versionado; o que for aprovado vai para evals/dataset.jsonl)

Como funciona:
1. Sorteia seções do corpus (semente fixa), no máximo uma por documento, ignorando seções
   curtas ou só de referências/links.
2. Para cada seção, pede ao LLM DUAS perguntas: uma "exata" (usa o termo técnico da seção) e
   uma "paráfrase" (evita as palavras do texto), mais fatos-chave curtos.
3. Fatos-chave que não aparecem literalmente na seção são descartados.

Viés conhecido: perguntas geradas a partir do texto tendem a repetir as palavras dele,
o que favorece o BM25. A pergunta de paráfrase existe para medir justamente isso.
"""

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from grounded_rag.config import get_settings
from grounded_rag.generation.llm import OllamaChat
from grounded_rag.ingest.clean import clean_markdown
from grounded_rag.ingest.pipeline import find_documents
from grounded_rag.ingest.readers import LOCATION_SEP, read_document

SEED = 7
MIN_WORDS = 80
SKIP_TITLES = ("reference", "related", "author", "resource", "tool", "further reading", "credit")

PROMPT = """You write evaluation questions for a search system over security documentation.
Given ONE section of a document, write:
- "exact_question": a question a developer would ask, using the specific technical terms
  that appear in the section.
- "paraphrase_question": a question with the SAME answer, but phrased in everyday words,
  avoiding the distinctive words of the section (no copied phrases, no header names).
- "key_facts": 1 to 3 very short phrases (1-5 words) copied EXACTLY from the section that a
  correct answer must contain (numbers, names, settings).
Both questions must be answerable from this section alone and must not mention "the section",
"the document" or "the cheat sheet"."""

SCHEMA = {
    "type": "object",
    "properties": {
        "exact_question": {"type": "string"},
        "paraphrase_question": {"type": "string"},
        "key_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["exact_question", "paraphrase_question", "key_facts"],
}


def candidate_sections(data_dir: Path) -> list:
    """Uma seção sorteada por documento, entre as que têm conteúdo suficiente."""
    rng = random.Random(SEED)
    picked = []
    for path in find_documents(data_dir):
        segments = [replace(s, text=clean_markdown(s.text)) for s in read_document(path, data_dir)]
        good = [
            s
            for s in segments
            if len(s.text.split()) >= MIN_WORDS
            and LOCATION_SEP in s.location  # não a introdução solta antes do 1º subtítulo
            and not any(t in s.location.split(LOCATION_SEP)[-1].lower() for t in SKIP_TITLES)
        ]
        if good:
            picked.append(rng.choice(good))
    rng.shuffle(picked)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default="evals/candidates.jsonl")
    args = parser.parse_args()

    settings = get_settings()
    llm = OllamaChat(str(settings.ollama_base_url), args.model or settings.llm_model)
    sections = candidate_sections(settings.data_dir)[: args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, section in enumerate(sections, start=1):
            messages = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Section ({section.location}):\n\n{section.text}"},
            ]
            result = llm.complete_json(messages, SCHEMA)
            text = section.text.lower()
            facts = [k for k in result["key_facts"] if k.strip() and k.lower() in text]
            row = {
                "source": section.source,
                "section": section.location,
                "exact_question": result["exact_question"].strip(),
                "paraphrase_question": result["paraphrase_question"].strip(),
                "key_facts": facts,
                "section_text": section.text[:1500],  # para a revisão humana
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"\r{i}/{len(sections)}", end="", flush=True)
    print(f"\n{len(sections)} candidatos em {out}")


if __name__ == "__main__":
    main()
