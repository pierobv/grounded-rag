"""Valida o juiz de fidelidade contra rótulos humanos.

Uso:
  uv run python scripts/judge_agreement.py sample D2-qwen7b   # sorteia 30 vereditos para rotular
  (edite evals/judge_labels.jsonl preenchendo "human": true/false em cada linha)
  uv run python scripts/judge_agreement.py score              # concordância juiz x humano

Por que: o juiz é um LLM. Se ele concorda pouco com um humano, a métrica de fidelidade
não vale muito, e o relatório precisa dizer isso.
"""

import json
import random
import sys
from pathlib import Path

RESULTS = Path("evals/results")
LABELS = Path("evals/judge_labels.jsonl")
SAMPLE_SIZE = 30
SEED = 11


def sample(experiment: str) -> None:
    result = json.loads((RESULTS / f"{experiment}.json").read_text(encoding="utf-8"))
    verdicts = [
        {
            "question_id": q["id"],
            "claim": v["claim"],
            "excerpts": v["excerpts"],
            "judge": v["supported"],
            "judge_reason": v["reason"],
            "human": None,
        }
        for q in result["questions"]
        for v in q.get("verdicts", [])
    ]
    rng = random.Random(SEED)
    chosen = rng.sample(verdicts, min(SAMPLE_SIZE, len(verdicts)))
    LABELS.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in chosen))
    print(f"{len(chosen)} vereditos em {LABELS}; preencha o campo 'human'.")


def score() -> None:
    rows = [json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()]
    labeled = [r for r in rows if r["human"] is not None]
    agree = sum(r["judge"] == r["human"] for r in labeled)
    # Matriz de confusão: o erro mais grave é o juiz dizer "sustentado" quando não é.
    false_support = sum(r["judge"] and not r["human"] for r in labeled)
    false_reject = sum(not r["judge"] and r["human"] for r in labeled)
    print(f"Rotulados: {len(labeled)}/{len(rows)}")
    print(f"Concordância: {agree}/{len(labeled)} = {agree / len(labeled):.0%}")
    print(f"Juiz aprovou o que o humano rejeitou: {false_support}")
    print(f"Juiz rejeitou o que o humano aprovou: {false_reject}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["sample"]:
        sample(sys.argv[2])
    elif sys.argv[1:2] == ["score"]:
        score()
    else:
        sys.exit(__doc__)
