"""Runner: executa um experimento (uma configuração completa) sobre o dataset.

Para cada pergunta:
1. busca os RETRIEVE_K melhores trechos -> métricas de busca (recall@k, MRR);
2. se o experimento tem LLM: gera a resposta com os k primeiros desses mesmos trechos
   -> abstenção, citações, fatos-chave, latência e, com juiz, fidelidade.

O resultado vai para evals/results/<nome>.json com tudo o que é preciso para reproduzi-lo:
configuração, commit do código, commit do corpus, versão do prompt e o detalhe por pergunta.
"""

import json
import subprocess
import time
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from grounded_rag.config import Settings
from grounded_rag.evaluation import metrics
from grounded_rag.evaluation.dataset import EvalQuestion
from grounded_rag.evaluation.judge import Judge, judge_answer
from grounded_rag.generation.answer import Final, stream_answer
from grounded_rag.generation.prompt import PROMPT_VERSION
from grounded_rag.ingest.pipeline import ingest_folder, read_jsonl, write_jsonl
from grounded_rag.retrieval import factory
from grounded_rag.retrieval.base import SearchResult
from grounded_rag.retrieval.factory import ChunkingConfig

RETRIEVE_K = 20  # = CANDIDATES: mede também o teto do funil do reranking
MRR_K = 10


class Experiment(BaseModel):
    """Uma configuração completa. Campos ausentes no TOML usam [defaults] e depois estes."""

    name: str
    description: str = ""
    strategy: str = "structure"
    size: int = 200
    overlap: int = 40
    headings: bool = False
    mode: str = "hybrid"
    rerank: bool = True
    embedding_model: str = "nomic-embed-text"
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str | None = None  # None = só busca (rápido)
    judge: bool = False  # julgar fidelidade das citações (lento)
    k: int = 5  # trechos enviados ao LLM

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(self.strategy, self.size, self.overlap, self.headings)

    def settings(self, base: Settings) -> Settings:
        return base.model_copy(
            update={
                "embedding_model": self.embedding_model,
                "embed_doc_prefix": self.embed_doc_prefix,
                "embed_query_prefix": self.embed_query_prefix,
                "rerank_model": self.rerank_model,
                "candidates": RETRIEVE_K,
            }
        )


def load_experiments(path: Path) -> list[Experiment]:
    """TOML: uma tabela [defaults] e uma lista [[experiment]] que a sobrescreve."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    experiments = [Experiment(**(defaults | entry)) for entry in data.get("experiment", [])]
    names = [e.name for e in experiments]
    if len(names) != len(set(names)):
        raise ValueError("Nomes de experimento repetidos")
    return experiments


class _Precomputed:
    """Buscador que devolve resultados já calculados: a geração usa os mesmos trechos da
    avaliação de busca, sem buscar (e reordenar) de novo."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, query: str, k: int) -> list[SearchResult]:
        return self.results[:k]


def ensure_index(settings: Settings, exp: Experiment) -> None:
    """Gera trechos e índice vetorial da configuração, se ainda não existirem."""
    config = exp.chunking
    path = factory.chunks_path(settings, config)
    if path.exists():
        chunks = read_jsonl(path)
    else:
        chunks = ingest_folder(
            settings.data_dir, config.strategy, config.size, config.overlap, config.headings
        )
        write_jsonl(chunks, path)
    if exp.mode in ("vector", "hybrid"):
        factory.open_vector_index(settings, config).sync(chunks)  # só embedda o que falta


def _hits(results: list[SearchResult]) -> list[metrics.Hit]:
    return [(r.chunk.source, r.chunk.location) for r in results]


def evaluate_question(
    q: EvalQuestion,
    results: list[SearchResult],
    search_ms: float,
    exp: Experiment,
    llm,
) -> dict:
    hits = _hits(results)
    row: dict = {
        "id": q.id,
        "type": q.type,
        "lang": q.lang,
        "answerable": q.answerable,
        "search_ms": round(search_ms, 1),
        "top_score": results[0].score if results else None,
        "top5": [list(h) for h in hits[:5]],
    }
    if q.answerable:
        row |= {
            "first_relevant_rank": metrics.first_relevant_rank(hits, q.relevant),
            "recall@1": metrics.recall_at_k(hits, q.relevant, 1),
            "recall@5": metrics.recall_at_k(hits, q.relevant, 5),
            "recall@20": metrics.recall_at_k(hits, q.relevant, 20),
            "mrr@10": metrics.reciprocal_rank(hits, q.relevant, MRR_K),
        }

    if llm is None:
        return row

    events = list(stream_answer(q.question, _Precomputed(results), llm, exp.k))
    final = next(e for e in events if isinstance(e, Final))
    answer, stats = final.answer, final.stats
    cited = [(c.source, c.location) for c in answer.citations]
    row |= {
        "abstained": answer.abstained,
        "answer": answer.text,
        "citations": [c.number for c in answer.citations],
        "invalid_citations": answer.invalid_citations,
        "citation_precision": (
            metrics.citation_precision(cited, q.relevant) if q.answerable else None
        ),
        "key_fact_recall": (
            None if answer.abstained else metrics.key_fact_recall(answer.text, q.key_facts)
        ),
        "ttft_ms": stats.time_to_first_token_ms,
        "generation_ms": stats.total_ms,
        "prompt_tokens": stats.prompt_tokens,
    }
    return row


def judge_row(row: dict, results: list[SearchResult], exp: Experiment, judge: Judge) -> None:
    """Fidelidade de uma resposta já gerada (segunda passada, depois de toda a geração)."""
    if row["abstained"]:
        return
    verdicts, uncited = judge_answer(row["answer"], [r.chunk for r in results[: exp.k]], judge)
    row |= {
        "faithfulness": sum(v.supported for v in verdicts) / len(verdicts) if verdicts else None,
        "uncited_sentences": uncited,
        "verdicts": [v.__dict__ for v in verdicts],
    }


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def aggregate(rows: list[dict]) -> dict:
    """Médias com intervalo de confiança (bootstrap) e recortes por tipo de pergunta."""

    def summary(key: str, subset: list[dict]) -> dict | None:
        values = [r[key] for r in subset if r.get(key) is not None]
        if not values:
            return None
        low, high = metrics.bootstrap_ci(values)
        return {"mean": metrics.mean(values), "ci95": [low, high], "n": len(values)}

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    result: dict = {
        "retrieval": {
            key: summary(key, answerable) for key in ("recall@1", "recall@5", "recall@20", "mrr@10")
        },
        "retrieval_by_type": {
            group: summary("recall@5", [r for r in answerable if r[field] == group])
            for field, groups in (
                ("type", ("exact", "paraphrase", "multi")),
                ("lang", ("en", "pt")),
            )
            for group in groups
        },
        "search_ms_p50": _percentile([r["search_ms"] for r in rows], 0.5),
    }

    if rows and "abstained" in rows[0]:
        answered = [r for r in answerable if not r["abstained"]]
        result["generation"] = {
            # Duas taxas juntas: "sempre dizer não sei" acertaria 100% da primeira.
            "refusal_rate_unanswerable": summary(
                "refused", [{"refused": float(r["abstained"])} for r in unanswerable]
            ),
            "answer_rate_answerable": summary(
                "answered", [{"answered": float(not r["abstained"])} for r in answerable]
            ),
            "citation_precision": summary("citation_precision", answered),
            "key_fact_recall": summary("key_fact_recall", answered),
            "faithfulness": summary("faithfulness", answered),
            "invalid_citation_rate": summary(
                "invalid", [{"invalid": float(bool(r["invalid_citations"]))} for r in rows]
            ),
            "ttft_ms_p50": _percentile([r["ttft_ms"] for r in rows if r["ttft_ms"]], 0.5),
            "generation_ms_p50": _percentile([r["generation_ms"] for r in rows], 0.5),
            "generation_ms_p95": _percentile([r["generation_ms"] for r in rows], 0.95),
        }
    return result


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _corpus_commit(data_dir: Path) -> str | None:
    manifests = sorted(data_dir.glob("*/MANIFEST.json"))
    return json.loads(manifests[0].read_text())["commit"] if manifests else None


def run_experiment(
    exp: Experiment,
    base: Settings,
    questions: list[EvalQuestion],
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> dict:
    settings = exp.settings(base)
    ensure_index(settings, exp)
    retriever = factory.build_retriever(settings, exp.chunking, exp.mode, exp.rerank)
    llm = factory.make_llm(settings, exp.llm_model) if exp.llm_model else None
    judge = factory.make_judge(settings) if (llm and exp.judge) else None

    rows = []
    all_results = []
    for done, q in enumerate(questions, start=1):
        started = time.perf_counter()
        results = retriever.search(q.question, RETRIEVE_K)
        search_ms = (time.perf_counter() - started) * 1000
        rows.append(evaluate_question(q, results, search_ms, exp, llm))
        all_results.append(results)
        progress(done, len(questions))

    # O juiz roda numa segunda passada, depois de toda a geração: assim só um LLM fica na
    # memória por vez. Alternar gerador e juiz a cada pergunta fazia o Ollama trocar de
    # modelo ~200 vezes e, com 16 GB, chegou a falhar ao carregar (decisão 032).
    if judge is not None:
        for done, (row, results) in enumerate(zip(rows, all_results, strict=True), start=1):
            judge_row(row, results, exp, judge)
            progress(done, len(questions))

    return {
        "experiment": exp.model_dump(),
        "index": exp.chunking.index_name,
        "meta": {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "corpus_commit": _corpus_commit(settings.data_dir),
            "prompt_version": PROMPT_VERSION,
            "judge_model": settings.llm_model if judge else None,
            "n_questions": len(questions),
        },
        "aggregate": aggregate(rows),
        "questions": rows,
    }


def save_result(result: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result['experiment']['name']}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
