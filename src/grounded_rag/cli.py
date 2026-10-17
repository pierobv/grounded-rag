"""Interface de linha de comando: `grounded-rag ingest | search | ask | eval`.

O CLI é uma camada fina: lê argumentos, chama as funções do projeto e imprime o resultado.
`ask` e `eval` ganham implementação nas próximas etapas.
"""

import argparse
import sys
import time

from grounded_rag.config import Settings, get_settings
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

    ingest = sub.add_parser(
        "ingest", parents=[chunking], help="Divide os documentos em trechos e os indexa."
    )
    ingest.add_argument(
        "--no-index", action="store_true", help="Só gera os trechos, sem embeddings (sem Ollama)."
    )

    search = sub.add_parser(
        "search", parents=[chunking], help="Busca trechos relevantes (sem gerar resposta)."
    )
    search.add_argument("query", help="Texto da busca.")
    search.add_argument("--mode", choices=["vector", "bm25", "hybrid"], help="Tipo de busca.")
    search.add_argument("-k", type=int, help="Quantos trechos devolver.")
    # BooleanOptionalAction cria --rerank e --no-rerank; None = usar o valor do .env.
    search.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, help="Reordenar com cross-encoder."
    )

    ask = sub.add_parser("ask", help="Responde a uma pergunta citando fonte e localização.")
    ask.add_argument("question", help="Pergunta em linguagem natural.")

    sub.add_parser("eval", help="Roda a avaliação sobre evals/dataset.jsonl.")

    return parser


def resolve_chunking(args: argparse.Namespace, settings: Settings) -> ChunkingConfig:
    """Argumento da linha de comando tem prioridade; senão, vale o .env / padrão."""

    def pick(arg, default):
        return default if arg is None else arg

    return ChunkingConfig(
        strategy=pick(args.strategy, settings.chunk_strategy),
        size=pick(args.size, settings.chunk_size),
        overlap=pick(args.overlap, settings.chunk_overlap),
    )


def run_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    config = resolve_chunking(args, settings)

    if not settings.data_dir.is_dir():
        print(f"Pasta de dados não encontrada: {settings.data_dir}", file=sys.stderr)
        return 1

    try:
        chunks = ingest_folder(settings.data_dir, config.strategy, config.size, config.overlap)
    except ValueError as exc:  # ex.: overlap >= size passado pela linha de comando
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    output = chunks_path(settings, config)
    write_jsonl(chunks, output)

    stats = compute_stats(chunks)
    print(f"Estratégia: {config.strategy} (size={config.size}, overlap={config.overlap})")
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


COMMANDS = {"ingest": run_ingest, "search": run_search}


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
