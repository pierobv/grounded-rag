# Grounded RAG

Assistente de perguntas e respostas sobre documentos, rodando 100% local e com custo zero.
Toda resposta cita fonte e página; se os trechos recuperados não contêm a resposta,
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
uv run pytest           # testes
uv run grounded-rag --help
```
