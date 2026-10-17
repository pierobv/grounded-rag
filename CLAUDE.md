# Grounded RAG — contexto para o Claude Code

## O projeto
Assistente de perguntas e respostas sobre documentos (RAG), rodando 100% local e com custo zero.
O objetivo principal é ser um projeto de portfólio: o diferencial é a **avaliação** — cada decisão
(chunking, busca, reranking, modelo) é medida com métricas e documentada.

O autor (Piero) é desenvolvedor .NET/Python/React e está aprendendo AI engineering.
**Explique as decisões e o código que você escrever.** Ele precisa entender cada linha para
defender o projeto em entrevistas. Não esconda lógica central atrás de frameworks prontos
(nada de LangChain/LlamaIndex): chunking, busca híbrida, fusão de rankings e métricas são
implementados no próprio projeto.

## Stack
- Python 3.12+, gerenciado com `uv`
- Modelos via Ollama (API HTTP em `OLLAMA_BASE_URL`): um LLM de 7–8B, um de 3B para desenvolvimento, um modelo de embedding
- Chroma em modo persistente (sem servidor)
- BM25 com `rank-bm25`; reranking com cross-encoder via `sentence-transformers`
- `pymupdf` para PDFs
- FastAPI para a API; React em `web/` para a interface
- `pytest`, `ruff`, GitHub Actions, Semgrep

## Estrutura
- `src/grounded_rag/config.py` — toda configuração vem de variáveis de ambiente (ver `.env.example`)
- `src/grounded_rag/ingest/` — leitura, limpeza, chunking (fixo com sobreposição e por estrutura)
- `src/grounded_rag/retrieval/` — vetorial, BM25, fusão RRF, reranking (cada um ligável por config)
- `src/grounded_rag/generation/` — prompt, chamada ao Ollama, citações validadas com Pydantic
- `src/grounded_rag/api/` — FastAPI (`/ask` com streaming, `/health`)
- `src/grounded_rag/cli.py` — comandos `ingest`, `search`, `ask`, `eval`
- `evals/` — `dataset.jsonl`, métricas, runner e `results/`
- `tests/`, `web/`, `docs/decisions.md`

## Comandos
- Instalar: `uv sync`
- Testes: `uv run pytest`
- Lint e formatação: `uv run ruff check . && uv run ruff format .`
- CLI: `uv run grounded-rag ingest | search "..." | ask "..." | eval`

## Regras
- Trabalhamos **uma etapa por vez**. Mostre o plano antes de escrever código e espere aprovação.
- Escreva testes junto com cada funcionalidade. Nos testes, simule o Ollama: o CI não tem modelo.
- Nunca leia, copie ou faça commit de nada em `data/private/` (material privado do mestrado).
  O corpus público fica em `data/public/` (OWASP Cheat Sheets).
- Nunca coloque chaves, tokens ou endereços pessoais no código; use `.env` (ignorado pelo Git).
- Toda resposta do assistente deve citar fonte e página; se os trechos não contêm a resposta, ele diz que não sabe.
- O endereço do Ollama nunca é fixo no código: sempre `OLLAMA_BASE_URL`
  (localhost no Mac; o PC via Tailscale para avaliações longas).
- Ao terminar uma etapa, resuma o que mudou e sugira a mensagem de commit.
- Registre decisões relevantes em `docs/decisions.md` (o quê, alternativas, por quê).
