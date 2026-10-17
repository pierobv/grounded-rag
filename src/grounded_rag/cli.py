"""Interface de linha de comando: `grounded-rag ingest | search | ask | eval`.

O CLI é uma camada fina: lê argumentos, chama as funções do projeto e imprime o resultado.
`search`, `ask` e `eval` ganham implementação nas próximas etapas.
"""

import argparse
import sys

from grounded_rag.config import get_settings
from grounded_rag.ingest.chunking import STRATEGIES
from grounded_rag.ingest.pipeline import compute_stats, ingest_folder, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded-rag",
        description="Perguntas e respostas sobre documentos, com citação de fonte e localização.",
    )
    # Cada subcomando vira um "sub-parser" com seus próprios argumentos.
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Lê, limpa e divide em trechos os documentos.")
    # Sem valor padrão aqui: None significa "use o valor da configuração (.env)".
    ingest.add_argument("--strategy", choices=sorted(STRATEGIES), help="Estratégia de chunking.")
    ingest.add_argument("--size", type=int, help="Tamanho do trecho, em palavras.")
    ingest.add_argument("--overlap", type=int, help="Sobreposição entre trechos, em palavras.")

    search = sub.add_parser("search", help="Busca trechos relevantes (sem gerar resposta).")
    search.add_argument("query", help="Texto da busca.")

    ask = sub.add_parser("ask", help="Responde a uma pergunta citando fonte e localização.")
    ask.add_argument("question", help="Pergunta em linguagem natural.")

    sub.add_parser("eval", help="Roda a avaliação sobre evals/dataset.jsonl.")

    return parser


def run_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    strategy = args.strategy or settings.chunk_strategy
    size = settings.chunk_size if args.size is None else args.size
    overlap = settings.chunk_overlap if args.overlap is None else args.overlap

    if not settings.data_dir.is_dir():
        print(f"Pasta de dados não encontrada: {settings.data_dir}", file=sys.stderr)
        return 1

    try:
        chunks = ingest_folder(settings.data_dir, strategy, size, overlap)
    except ValueError as exc:  # ex.: overlap >= size passado pela linha de comando
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    output = settings.processed_dir / "chunks.jsonl"
    write_jsonl(chunks, output)

    stats = compute_stats(chunks)
    print(f"Estratégia: {strategy} (size={size}, overlap={overlap})")
    print(f"Documentos: {stats.documents}  Trechos: {stats.chunks}")
    print(
        f"Palavras por trecho: média {stats.avg_words:.0f}, "
        f"mín {stats.min_words}, máx {stats.max_words}"
    )
    print(f"Salvo em {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # argv=None faz o argparse ler sys.argv; nos testes passamos a lista diretamente.
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        return run_ingest(args)
    print(f"Comando '{args.command}' ainda não implementado.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
