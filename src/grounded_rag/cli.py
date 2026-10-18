"""Interface de linha de comando: `grounded-rag ingest | search | ask | eval`.

O CLI é uma camada fina: lê argumentos, chama as funções do projeto e imprime o resultado.
`eval` roda os experimentos de evals/experiments.toml e gera o relatório.
"""

import argparse
import sys
import time

from grounded_rag.config import Settings, get_settings
from grounded_rag.generation.answer import Final, Retrieved, TextDelta, stream_answer
from grounded_rag.ingest.chunking import STRATEGIES
from grounded_rag.ingest.pipeline import compute_stats, ingest_folder, write_jsonl
from grounded_rag.retrieval import factory
from grounded_rag.retrieval.base import SearchResult
from grounded_rag.retrieval.embeddings import OllamaError
from grounded_rag.retrieval.factory import ChunkingConfig, IndexNotReadyError, chunks_path

SNIPPET_CHARS = 200


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded-rag",
        description="Perguntas e respostas sobre documentos, com citação de fonte e localização.",
    )
    # Cada subcomando vira um "sub-parser" com seus próprios argumentos.
    sub = parser.add_subparsers(dest="command", required=True)

    # Opções de chunking compartilhadas por ingest e search ("parent parser"): a busca
    # precisa usar exatamente o índice gerado com a mesma configuração.
    # Sem valor padrão aqui: None significa "use o valor da configuração (.env)".
    chunking = argparse.ArgumentParser(add_help=False)
    chunking.add_argument("--strategy", choices=sorted(STRATEGIES), help="Estratégia de chunking.")
    chunking.add_argument("--size", type=int, help="Tamanho do trecho, em palavras.")
    chunking.add_argument("--overlap", type=int, help="Sobreposição entre trechos, em palavras.")
    chunking.add_argument(
        "--headings",
        action=argparse.BooleanOptionalAction,
        help="Põe o caminho de títulos no início de cada trecho.",
    )

    ingest = sub.add_parser(
        "ingest", parents=[chunking], help="Divide os documentos em trechos e os indexa."
    )
    ingest.add_argument(
        "--no-index", action="store_true", help="Só gera os trechos, sem embeddings (sem Ollama)."
    )

    # Opções de busca compartilhadas por search e ask.
    retrieval = argparse.ArgumentParser(add_help=False)
    retrieval.add_argument("--mode", choices=["vector", "bm25", "hybrid"], help="Tipo de busca.")
    retrieval.add_argument("-k", type=int, help="Quantos trechos buscar.")
    # BooleanOptionalAction cria --rerank e --no-rerank; None = usar o valor do .env.
    retrieval.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, help="Reordenar com cross-encoder."
    )

    search = sub.add_parser(
        "search",
        parents=[chunking, retrieval],
        help="Busca trechos relevantes (sem gerar resposta).",
    )
    search.add_argument("query", help="Texto da busca.")

    ask = sub.add_parser(
        "ask",
        parents=[chunking, retrieval],
        help="Responde a uma pergunta citando fonte e localização.",
    )
    ask.add_argument("question", help="Pergunta em linguagem natural.")
    model = ask.add_mutually_exclusive_group()
    model.add_argument("--dev", action="store_true", help="Usa o modelo pequeno (DEV_LLM_MODEL).")
    model.add_argument("--model", help="Usa este modelo do Ollama.")
    ask.add_argument(
        "--show-context", action="store_true", help="Mostra os trechos enviados ao modelo."
    )

    evaluate = sub.add_parser("eval", help="Roda a avaliação sobre evals/dataset.jsonl.")
    evaluate.add_argument(
        "--experiments", help="Nomes separados por vírgula (padrão: todos do arquivo)."
    )
    evaluate.add_argument(
        "--file", default="evals/experiments.toml", help="Arquivo de experimentos."
    )
    evaluate.add_argument("--dataset", default="evals/dataset.jsonl", help="Dataset de perguntas.")
    evaluate.add_argument("--results", default="evals/results", help="Pasta dos resultados.")
    evaluate.add_argument(
        "--retrieval-only", action="store_true", help="Ignora o LLM (só métricas de busca)."
    )
    evaluate.add_argument(
        "--report-only", action="store_true", help="Só regenera o relatório a partir dos JSON."
    )

    return parser


def resolve_chunking(args: argparse.Namespace, settings: Settings) -> ChunkingConfig:
    """Argumento da linha de comando tem prioridade; senão, vale o .env / padrão."""

    def pick(arg, default):
        return default if arg is None else arg

    return ChunkingConfig(
        strategy=pick(args.strategy, settings.chunk_strategy),
        size=pick(args.size, settings.chunk_size),
        overlap=pick(args.overlap, settings.chunk_overlap),
        headings=pick(args.headings, settings.chunk_headings),
    )


def run_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    config = resolve_chunking(args, settings)

    if not settings.data_dir.is_dir():
        print(f"Pasta de dados não encontrada: {settings.data_dir}", file=sys.stderr)
        return 1

    try:
        chunks = ingest_folder(
            settings.data_dir, config.strategy, config.size, config.overlap, config.headings
        )
    except ValueError as exc:  # ex.: overlap >= size passado pela linha de comando
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    output = chunks_path(settings, config)
    write_jsonl(chunks, output)

    stats = compute_stats(chunks)
    print(f"Índice: {config.index_name}")
    print(f"Documentos: {stats.documents}  Trechos: {stats.chunks}")
    print(
        f"Palavras por trecho: média {stats.avg_words:.0f}, "
        f"mín {stats.min_words}, máx {stats.max_words}"
    )
    print(f"Salvo em {output}")

    if args.no_index:
        return 0

    started = time.perf_counter()
    try:
        sync = factory.open_vector_index(settings, config).sync(chunks)
    except OllamaError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(
        f"Índice vetorial: {sync.added} novos, {sync.removed} removidos, "
        f"{sync.kept} mantidos ({elapsed:.1f}s)"
    )
    return 0


def describe_origins(result: SearchResult) -> str:
    """{"vector": 3, "bm25": 1} -> "vetorial #3 · bm25 #1"."""
    labels = {"vector": "vetorial", "bm25": "bm25", "before_rerank": "antes do rerank"}
    return " · ".join(f"{labels.get(name, name)} #{rank}" for name, rank in result.origins.items())


def run_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    config = resolve_chunking(args, settings)
    mode = args.mode or settings.retrieval_mode
    k = args.k or settings.top_k
    rerank = settings.rerank if args.rerank is None else args.rerank

    try:
        retriever = factory.build_retriever(settings, config, mode, rerank)
        started = time.perf_counter()
        results = retriever.search(args.query, k)
        elapsed = time.perf_counter() - started
    except (IndexNotReadyError, OllamaError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    label = f"{mode} + rerank" if rerank else mode
    print(
        f"Busca {label} em {config.index_name}: {len(results)} resultado(s) "
        f"em {elapsed * 1000:.0f} ms\n"
    )
    for r in results:
        snippet = " ".join(r.chunk.text.split())[:SNIPPET_CHARS]
        print(f"{r.rank}. [{r.score:.3f}] {r.chunk.source} — {r.chunk.location}")
        if r.origins:
            print(f"   ({describe_origins(r)})")
        print(f"   {snippet}\n")
    return 0


def run_ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    config = resolve_chunking(args, settings)
    mode = args.mode or settings.retrieval_mode
    k = args.k or settings.top_k
    rerank = settings.rerank if args.rerank is None else args.rerank
    model = args.model or (settings.dev_llm_model if args.dev else settings.llm_model)
    # O portão de confiança usa a escala do cross-encoder: só vale com reranking ligado.
    min_score = settings.min_rerank_score if rerank else None

    streamed = False
    try:
        retriever = factory.build_retriever(settings, config, mode, rerank)
        llm = factory.make_llm(settings, model)
        events = stream_answer(
            args.question, retriever, llm, k, min_score, settings.abstain_message
        )
        for event in events:
            if isinstance(event, Retrieved) and args.show_context:
                print("Trechos enviados ao modelo:")
                for n, r in enumerate(event.results, start=1):
                    print(f"  [{n}] ({r.score:.3f}) {r.chunk.source} — {r.chunk.location}")
                print()
            elif isinstance(event, TextDelta):
                # flush=True: mostra cada pedaço assim que chega (streaming de verdade).
                print(event.text, end="", flush=True)
                streamed = True
            elif isinstance(event, Final):
                print_final(event, streamed)
    except (IndexNotReadyError, OllamaError) as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        return 1
    return 0


def print_final(final: Final, streamed: bool) -> None:
    answer, stats = final.answer, final.stats
    if streamed:
        print("\n")

    if answer.abstained:
        if streamed:
            # O modelo escreveu algo, mas sem nenhuma citação válida: a resposta é descartada.
            print("⚠ Resposta descartada: nenhuma afirmação tinha citação válida.")
        print(answer.text)
    else:
        if answer.invalid_citations:
            invalid = ", ".join(f"[{n}]" for n in answer.invalid_citations)
            print(f"⚠ Citações para trechos inexistentes removidas: {invalid}\n")
        print("Fontes:")
        for c in answer.citations:
            print(f"  [{c.number}] {c.source} — {c.location}")

    parts = [stats.model, f"busca {stats.retrieval_ms:.0f} ms"]
    if stats.time_to_first_token_ms is None:
        parts.append("LLM não chamado")
    else:
        parts += [
            f"1º token {stats.time_to_first_token_ms / 1000:.1f} s",
            f"total {stats.total_ms / 1000:.1f} s",
            f"{stats.completion_tokens} tokens ({stats.tokens_per_second:.0f} tok/s)",
            f"prompt {stats.prompt_tokens} tokens",
        ]
    print(f"\n({' · '.join(parts)})")
    if stats.context_near_limit:
        print("⚠ O prompt está perto do limite de contexto (NUM_CTX): trechos podem ser cortados.")


def run_eval(args: argparse.Namespace) -> int:
    # Imports locais: a avaliação puxa módulos que os outros comandos não precisam.
    from pathlib import Path

    from grounded_rag.evaluation.dataset import load_dataset, unknown_sections
    from grounded_rag.evaluation.report import write_report
    from grounded_rag.evaluation.runner import load_experiments, run_experiment, save_result

    results_dir = Path(args.results)
    if args.report_only:
        print(f"Relatório: {write_report(results_dir)}")
        return 0

    settings = get_settings()
    questions = load_dataset(Path(args.dataset))
    if problems := unknown_sections(questions, settings.data_dir):
        print("Seções rotuladas que não existem no corpus:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    experiments = load_experiments(Path(args.file))
    if args.experiments:
        wanted = set(args.experiments.split(","))
        if missing := wanted - {e.name for e in experiments}:
            print(f"Experimentos não encontrados: {', '.join(sorted(missing))}", file=sys.stderr)
            return 1
        experiments = [e for e in experiments if e.name in wanted]
    if args.retrieval_only:
        experiments = [e.model_copy(update={"llm_model": None}) for e in experiments]

    print(f"{len(questions)} perguntas · {len(experiments)} experimento(s)")
    for exp in experiments:
        started = time.perf_counter()

        def progress(done: int, total: int, name: str = exp.name) -> None:
            print(f"\r  {name}: {done}/{total}", end="", flush=True)

        try:
            result = run_experiment(exp, settings, questions, progress)
        except (IndexNotReadyError, OllamaError) as exc:
            print(f"\nErro em {exp.name}: {exc}", file=sys.stderr)
            return 1
        path = save_result(result, results_dir)
        recall = result["aggregate"]["retrieval"]["recall@5"]
        recall_txt = f"R@5 {recall['mean']:.0%}" if recall else ""
        print(f"  {recall_txt} · {time.perf_counter() - started:.0f}s → {path}")

    print(f"Relatório: {write_report(results_dir)}")
    return 0


COMMANDS = {"ingest": run_ingest, "search": run_search, "ask": run_ask, "eval": run_eval}


def main(argv: list[str] | None = None) -> int:
    # argv=None faz o argparse ler sys.argv; nos testes passamos a lista diretamente.
    args = build_parser().parse_args(argv)
    command = COMMANDS.get(args.command)
    if command is None:
        print(f"Comando '{args.command}' ainda não implementado.", file=sys.stderr)
        return 1
    return command(args)


if __name__ == "__main__":
    sys.exit(main())
