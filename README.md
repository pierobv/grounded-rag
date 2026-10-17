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
uv run grounded-rag ingest              # gera os trechos e o índice vetorial (Chroma)
uv run grounded-rag search "How do I prevent CSRF?"   # padrão: híbrido (RRF) + reranking
uv run grounded-rag search "..." --mode bm25 --no-rerank   # modos: vector | bm25 | hybrid
uv run pytest           # testes (offline; o Ollama é simulado)
uv run pytest -m ollama # testes de integração com o Ollama real
uv run pytest -m models # testes com o cross-encoder real (baixa o modelo)
uv run grounded-rag --help
```

## Corpus

O corpus público são os [OWASP Cheat Sheets](https://github.com/OWASP/CheatSheetSeries),
© OWASP Foundation, licenciados sob [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Os arquivos não são versionados aqui: `scripts/fetch_corpus.py` os baixa em um commit fixo, e
`data/public/owasp/MANIFEST.json` registra o commit e o hash de cada arquivo.
