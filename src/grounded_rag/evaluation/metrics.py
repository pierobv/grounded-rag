"""Métricas de avaliação, implementadas à mão (cada uma tem teste com valores calculados à mão).

Busca (por pergunta respondível):
- recall@k: fração das seções relevantes que aparecem nos k primeiros trechos.
  Para pergunta de uma seção só, é 1 ou 0 ("acertou ou não no top k").
- MRR@k: 1 / posição do primeiro trecho relevante (0 se não aparece até k).
  Recompensa achar cedo: 1º lugar vale 1, 2º vale 0,5, 5º vale 0,2.

Resposta:
- key_fact_recall: fração dos fatos-chave presentes no texto da resposta.
- citation_precision: fração dos trechos citados que pertencem a uma seção relevante.

Incerteza:
- bootstrap_ci: intervalo de confiança da média por reamostragem das perguntas.
"""

import random
from collections.abc import Sequence

from grounded_rag.evaluation.dataset import RelevantSection, section_matches
from grounded_rag.ingest.chunking import LOCATION_LIST_SEP

Hit = tuple[str, str]  # (source, location) de um trecho recuperado ou citado


def matching_sections(hit: Hit, relevant: Sequence[RelevantSection]) -> set[int]:
    """Índices das seções relevantes que o trecho cobre.

    Um trecho `fixed` pode atravessar seções ("A; B"): basta uma das partes estar dentro
    da seção esperada.
    """
    source, location = hit
    parts = location.split(LOCATION_LIST_SEP)
    return {
        i
        for i, rel in enumerate(relevant)
        if rel.source == source and any(section_matches(p, rel.section) for p in parts)
    }


def is_relevant(hit: Hit, relevant: Sequence[RelevantSection]) -> bool:
    return bool(matching_sections(hit, relevant))


def first_relevant_rank(hits: Sequence[Hit], relevant: Sequence[RelevantSection]) -> int | None:
    """Posição (a partir de 1) do primeiro trecho relevante, ou None se nenhum for."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, relevant):
            return rank
    return None


def recall_at_k(hits: Sequence[Hit], relevant: Sequence[RelevantSection], k: int) -> float:
    covered: set[int] = set()
    for hit in hits[:k]:
        covered |= matching_sections(hit, relevant)
    return len(covered) / len(relevant)


def reciprocal_rank(hits: Sequence[Hit], relevant: Sequence[RelevantSection], k: int) -> float:
    rank = first_relevant_rank(hits[:k], relevant)
    return 1.0 / rank if rank else 0.0


def key_fact_recall(text: str, facts: Sequence[str]) -> float | None:
    """Comparação sem diferenciar maiúsculas. None quando a pergunta não tem fatos-chave."""
    if not facts:
        return None
    lowered = text.lower()
    return sum(fact.lower() in lowered for fact in facts) / len(facts)


def citation_precision(cited: Sequence[Hit], relevant: Sequence[RelevantSection]) -> float | None:
    """None quando não há citações (ex.: abstenção)."""
    if not cited:
        return None
    return sum(is_relevant(hit, relevant) for hit in cited) / len(cited)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Intervalo de confiança da média por bootstrap percentil.

    Ideia: o dataset é uma amostra das perguntas possíveis. Sorteamos, com reposição,
    muitos "datasets alternativos" do mesmo tamanho e vemos quanto a média varia.
    Com ~70 perguntas, diferenças de poucos pontos entre configurações costumam ficar
    dentro dessa variação, e o intervalo deixa isso visível.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)  # semente fixa: o mesmo dataset gera sempre o mesmo intervalo
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    tail = (1 - confidence) / 2
    return means[int(tail * n_resamples)], means[int((1 - tail) * n_resamples) - 1]


def calibrate_threshold(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Limite que melhor separa scores de perguntas respondíveis (positivos) e não (negativos).

    Testa cada score observado como limite e escolhe o de maior acurácia balanceada:
    média entre "respondíveis acima do limite" e "não respondíveis abaixo dele".
    Balanceada porque há bem menos perguntas fora do corpus que respondíveis.
    """
    best_threshold, best_accuracy = 0.0, -1.0
    for threshold in sorted({*positives, *negatives}):
        kept = sum(s >= threshold for s in positives) / len(positives)
        rejected = sum(s < threshold for s in negatives) / len(negatives)
        accuracy = (kept + rejected) / 2
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy
    return best_threshold
