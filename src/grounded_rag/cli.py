"""Interface de linha de comando: `grounded-rag ingest | search | ask | eval`.

Nesta etapa os comandos são só o esqueleto; cada um ganha implementação na sua etapa.
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded-rag",
        description="Perguntas e respostas sobre documentos, com citação de fonte e página.",
    )
    # Cada subcomando vira um "sub-parser" com seus próprios argumentos.
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Lê, limpa e indexa os documentos de data/public.")

    search = sub.add_parser("search", help="Busca trechos relevantes (sem gerar resposta).")
    search.add_argument("query", help="Texto da busca.")

    ask = sub.add_parser("ask", help="Responde a uma pergunta citando fonte e página.")
    ask.add_argument("question", help="Pergunta em linguagem natural.")

    sub.add_parser("eval", help="Roda a avaliação sobre evals/dataset.jsonl.")

    return parser


def main(argv: list[str] | None = None) -> int:
    # argv=None faz o argparse ler sys.argv; nos testes passamos a lista diretamente.
    args = build_parser().parse_args(argv)
    print(f"Comando '{args.command}' ainda não implementado.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
