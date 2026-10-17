# Grounded RAG

Assistente de perguntas e respostas sobre documentos, rodando 100% local e com custo zero.
Toda resposta cita fonte e localização (página ou seção); se os trechos recuperados não contêm a resposta,
o assistente diz que não sabe.

O foco do projeto é a **avaliação**: cada decisão (chunking, busca, reranking, modelo) é
medida com métricas e registrada em [`docs/decisions.md`](docs/decisions.md).

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) com um LLM e um modelo de embedding

## Começando

```bash
uv sync                 # cria o ambiente e instala as dependências
cp .env.example .env    # ajuste OLLAMA_BASE_URL
uv run python scripts/fetch_corpus.py   # baixa o corpus (OWASP Cheat Sheets)
uv run grounded-rag ingest              # gera data/processed/chunks.jsonl
uv run pytest           # testes
uv run grounded-rag --help
```

## Corpus

O corpus público são os [OWASP Cheat Sheets](https://github.com/OWASP/CheatSheetSeries),
© OWASP Foundation, licenciados sob [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Os arquivos não são versionados aqui: `scripts/fetch_corpus.py` os baixa em um commit fixo, e
`data/public/owasp/MANIFEST.json` registra o commit e o hash de cada arquivo.
