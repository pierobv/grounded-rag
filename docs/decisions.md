# Registro de decisões

Cada decisão relevante: **o quê**, **alternativas consideradas** e **por quê**.

---

## 001 — `uv` como gerenciador de projeto

- **O quê:** `uv` para criar o ambiente, instalar dependências e travar versões (`uv.lock`).
- **Alternativas:** `pip` + `venv` + `requirements.txt`; Poetry.
- **Por quê:** uma ferramenta só para versão do Python, ambiente virtual e dependências;
  muito rápido; o `uv.lock` garante as mesmas versões no Mac, no PC e no CI.

## 002 — Python 3.12 fixo em `.python-version`

- **O quê:** o projeto roda em Python 3.12 (o `uv` baixa essa versão automaticamente).
- **Alternativas:** usar o Python do sistema (3.14).
- **Por quê:** bibliotecas de ML (PyTorch, que vem com `sentence-transformers`) costumam
  demorar a publicar pacotes para versões novas do Python. 3.12 é estável e bem suportado.

## 003 — Configuração com `pydantic-settings`

- **O quê:** uma classe `Settings` lê variáveis de ambiente e o `.env`, com tipos validados.
- **Alternativas:** `os.environ` direto; `python-dotenv` puro.
- **Por quê:** valida tipos (ex.: `OLLAMA_BASE_URL` precisa ser uma URL) e falha logo na
  inicialização, com mensagem clara, se algo obrigatório faltar. O Pydantic já entra no
  projeto para validar as citações.

## 004 — Dependências adicionadas na etapa em que são usadas

- **O quê:** a etapa 0 instala só `pydantic-settings`, `pytest` e `ruff`. Chroma, BM25,
  `sentence-transformers`, `pymupdf` e FastAPI entram nas suas etapas.
- **Alternativas:** instalar todo o stack de uma vez.
- **Por quê:** cada commit mostra junto a dependência e o código que a usa; o CI fica rápido
  enquanto o PyTorch (vários GB) ainda não é necessário.

## 005 — CLI com `argparse`

- **O quê:** subcomandos `ingest`, `search`, `ask` e `eval` com a biblioteca padrão.
- **Alternativas:** Typer, Click.
- **Por quê:** zero dependências e nenhuma "mágica": fácil de ler e explicar. O CLI é só
  uma camada fina sobre as funções do projeto.
