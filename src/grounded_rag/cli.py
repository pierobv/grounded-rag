"""Interface de linha de comando: `grounded-rag ingest | search | ask | eval`.

O CLI é uma camada fina: lê argumentos, chama as funções do projeto e imprime o resultado.
`ask` e `eval` ganham implementação nas próximas etapas.
"""

import argparse
import sys
import time
from dataclasses import dataclass

from grounded_rag.config import Settings, get_settings
from grounded_rag.ingest.chunking import STRATEGIES
from grounded_rag.ingest.pipeline import compute_stats, ingest_folder, read_jsonl, write_jsonl
from grounded_rag.retrieval.base import Retriever
from grounded_rag.retrieval.bm25 import BM25Index
from grounded_rag.retrieval.embeddings import Embedder, OllamaEmbedder, OllamaError
from grounded_rag.retrieval.vector import VectorIndex, collection_name

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
    search.add_argument("--mode", choices=["vector", "bm25"], help="Tipo de busca.")
    search.add_argument("-k", type=int, help="Quantos trechos devolver.")

    ask = sub.add_parser("ask", help="Responde a uma pergunta citando fonte e localização.")
    ask.add_argument("question", help="Pergunta em linguagem natural.")

    sub.add_parser("eval", help="Roda a avaliação sobre evals/dataset.jsonl.")

    return parser


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str
    size: int
    overlap: int

    @property
    def index_name(self) -> str:
        return f"{self.strategy}-{self.size}-{self.overlap}"


def resolve_chunking(args: argparse.Namespace, settings: Settings) -> ChunkingConfig:
    """Argumento da linha de comando tem prioridade; senão, vale o .env / padrão."""

    def pick(arg, default):
        return default if arg is None else arg

    return ChunkingConfig(
        strategy=pick(args.strategy, settings.chunk_strategy),
        size=pick(args.size, settings.chunk_size),
        overlap=pick(args.overlap, settings.chunk_overlap),
    )


def make_embedder(settings: Settings) -> Embedder:
    """Ponto único de criação do embedder (os testes o substituem por um falso)."""
    return OllamaEmbedder(
        base_url=str(settings.ollama_base_url),
        model=settings.embedding_model,
        doc_prefix=settings.embed_doc_prefix,
        query_prefix=settings.embed_query_prefix,
        batch_size=settings.embed_batch_size,
    )


def open_vector_index(settings: Settings, config: ChunkingConfig) -> VectorIndex:
    name = collection_name(config.index_name, settings.embedding_model)
    return VectorIndex(settings.chroma_dir, name, make_embedder(settings))


def chunks_path(settings: Settings, config: ChunkingConfig):
    return settings.processed_dir / f"chunks-{config.index_name}.jsonl"


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
        sync = open_vector_index(settings, config).sync(chunks)
    except OllamaError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(
        f"Índice vetorial: {sync.added} novos, {sync.removed} removidos, "
        f"{sync.kept} mantidos ({elapsed:.1f}s)"
    )
    return 0


def run_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    config = resolve_chunking(args, settings)
    mode = args.mode or settings.retrieval_mode
    k = args.k or settings.top_k

    retriever: Retriever
    if mode == "bm25":
        path = chunks_path(settings, config)
        if not path.exists():
            print(f"Trechos não encontrados ({path}). Rode `grounded-rag ingest`.", file=sys.stderr)
            return 1
        retriever = BM25Index(read_jsonl(path))
    else:
        retriever = open_vector_index(settings, config)
        if retriever.count() == 0:
            print("Índice vetorial vazio. Rode `grounded-rag ingest`.", file=sys.stderr)
            return 1

    try:
        results = retriever.search(args.query, k)
    except OllamaError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    print(f"Busca {mode} em {config.index_name}: {len(results)} resultado(s)\n")
    for r in results:
        snippet = " ".join(r.chunk.text.split())[:SNIPPET_CHARS]
        print(f"{r.rank}. [{r.score:.3f}] {r.chunk.source} — {r.chunk.location}")
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
