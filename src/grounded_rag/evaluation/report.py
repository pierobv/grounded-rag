"""Relatório: lê evals/results/*.json e gera evals/results/README.md com tabelas comparativas."""

import json
from pathlib import Path

from grounded_rag.evaluation.metrics import calibrate_threshold


def _fmt(summary: dict | None, pct: bool = True) -> str:
    """0.734 [0.65–0.81] -> "73 (65–81)"."""
    if not summary:
        return "–"
    scale = 100 if pct else 1
    low, high = summary["ci95"]
    return f"{summary['mean'] * scale:.0f} ({low * scale:.0f}–{high * scale:.0f})"


def _ms(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f}".replace(",", ".")


def load_results(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(results_dir.glob("*.json"))]


def retrieval_table(results: list[dict]) -> list[str]:
    lines = [
        "| Experimento | Índice | Busca | R@1 | R@5 | R@20 | MRR@10 | R@5 exato | R@5 paráfrase "
        "| R@5 multi | R@5 PT | Busca p50 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(
        results, key=lambda r: -(r["aggregate"]["retrieval"]["recall@5"] or {}).get("mean", 0)
    )
    for r in ordered:
        e, agg = r["experiment"], r["aggregate"]
        ret, by = agg["retrieval"], agg["retrieval_by_type"]
        search = e["mode"] + (" + rerank" if e["rerank"] else "")
        lines.append(
            f"| {e['name']} | {r['index']} · {e['embedding_model']} | {search} "
            f"| {_fmt(ret['recall@1'])} | **{_fmt(ret['recall@5'])}** | {_fmt(ret['recall@20'])} "
            f"| {_fmt(ret['mrr@10'])} | {_fmt(by['exact'])} | {_fmt(by['paraphrase'])} "
            f"| {_fmt(by['multi'])} | {_fmt(by['pt'])} | {_ms(agg['search_ms_p50'])} |"
        )
    return lines


def generation_table(results: list[dict]) -> list[str]:
    with_generation = [r for r in results if "generation" in r["aggregate"]]
    if not with_generation:
        return []
    lines = [
        "| Experimento | LLM | Responde (respondíveis) | Recusa (fora do corpus) "
        "| Precisão das citações | Fatos-chave | Fidelidade | Citação inválida "
        "| 1º token p50 (ms) | Total p50 / p95 (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in with_generation:
        g = r["aggregate"]["generation"]
        lines.append(
            f"| {r['experiment']['name']} | {r['experiment']['llm_model']} "
            f"| {_fmt(g['answer_rate_answerable'])} | {_fmt(g['refusal_rate_unanswerable'])} "
            f"| {_fmt(g['citation_precision'])} | {_fmt(g['key_fact_recall'])} "
            f"| {_fmt(g['faithfulness'])} | {_fmt(g['invalid_citation_rate'])} "
            f"| {_ms(g['ttft_ms_p50'])} | {_ms(g['generation_ms_p50'])} / "
            f"{_ms(g['generation_ms_p95'])} |"
        )
    return lines


def calibration_lines(results: list[dict]) -> list[str]:
    """Para experimentos com reranking: que limite de score separa respondíveis de não?"""
    lines = []
    for r in results:
        if not r["experiment"]["rerank"]:
            continue
        rows = [q for q in r["questions"] if q["top_score"] is not None]
        pos = [q["top_score"] for q in rows if q["answerable"]]
        neg = [q["top_score"] for q in rows if not q["answerable"]]
        if not pos or not neg:
            continue
        t = calibrate_threshold(pos, neg)
        kept = sum(s >= t for s in pos) / len(pos)
        rejected = sum(s < t for s in neg) / len(neg)
        lines.append(
            f"| {r['experiment']['name']} | {t:.2f} | {kept:.0%} | {rejected:.0%} "
            f"| {min(pos):.1f} / {sorted(pos)[len(pos) // 2]:.1f} "
            f"| {max(neg):.1f} / {sorted(neg)[len(neg) // 2]:.1f} |"
        )
    if not lines:
        return []
    return [
        "| Experimento | Limite | Respondíveis mantidas | Fora do corpus barradas "
        "| Score respondíveis (mín / mediana) | Score fora do corpus (máx / mediana) |",
        "|---|---|---|---|---|---|",
        *lines,
    ]


def build_report(results_dir: Path) -> str:
    results = load_results(results_dir)
    if not results:
        return "# Resultados\n\nNenhum resultado ainda. Rode `grounded-rag eval`.\n"
    meta = results[0]["meta"]
    parts = [
        "# Resultados da avaliação",
        "",
        "Gerado por `grounded-rag eval --report-only`. Não edite à mão.",
        "",
        f"- Perguntas: {meta['n_questions']} · corpus OWASP @ `{(meta['corpus_commit'] or '')[:7]}`"
        f" · prompt v{meta['prompt_version']}",
        "- Valores em %, com intervalo de confiança de 95% por bootstrap entre parênteses.",
        "- R@k = recall@k (seções relevantes no top k); recortes por tipo usam R@5.",
        "",
        "## Busca",
        "",
        *retrieval_table(results),
        "",
    ]
    if generation := generation_table(results):
        parts += ["## Resposta", "", *generation, ""]
    if calibration := calibration_lines(results):
        parts += [
            "## Calibração do portão de confiança (MIN_RERANK_SCORE)",
            "",
            "Score do melhor trecho após reranking; o limite maximiza a acurácia balanceada.",
            "",
            *calibration,
            "",
        ]
    return "\n".join(parts)


def write_report(results_dir: Path) -> Path:
    path = results_dir / "README.md"
    path.write_text(build_report(results_dir), encoding="utf-8")
    return path
