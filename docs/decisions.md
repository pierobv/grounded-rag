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

## 006 — Corpus público: OWASP Cheat Sheets em Markdown, em um commit fixo

- **O quê:** os `.md` originais do repositório `OWASP/CheatSheetSeries`, baixados por
  `scripts/fetch_corpus.py` em um commit fixo. Só o `MANIFEST.json` (commit + SHA-256 de cada
  arquivo) é versionado; `--check` confere se o corpus local é idêntico.
- **Alternativas:** gerar PDFs a partir do site; versionar os `.md` no repositório.
- **Por quê:** o Markdown é a fonte real (a OWASP não publica PDFs por cheat sheet; páginas de
  um PDF gerado por nós seriam arbitrárias). O commit fixo deixa as métricas comparáveis entre
  execuções. O manifesto já se pagou: o `ruff format` estava reformatando blocos de código
  dentro dos `.md`, e a divergência de hash revelou isso (agora `data/` é excluído do ruff).

## 007 — Citação por "fonte e localização"

- **O quê:** cada trecho carrega `source` e `location`: página para PDF (`p. 12`), caminho de
  títulos para Markdown (`XSS › Output Encoding`). A regra passa de "fonte e página" para
  "fonte e localização".
- **Alternativas:** só página (impossível em Markdown); número de linha (instável e pouco útil
  para o leitor).
- **Por quê:** a localização aponta para onde o leitor deve olhar em qualquer formato; uma seção
  costuma ser mais precisa que uma página inteira.

## 008 — Tamanho de trecho medido em palavras

- **O quê:** `CHUNK_SIZE` e `CHUNK_OVERLAP` em palavras (sequências sem espaço). Padrão 200/40.
- **Alternativas:** tokens do tokenizador do modelo de embedding; caracteres.
- **Por quê:** não amarra o chunking a um modelo (trocar o embedding não muda os trechos) e é
  fácil de explicar. Em inglês, 200 palavras ≈ 260–300 tokens, dentro do limite de 512 tokens
  típico de cross-encoders. Os valores são ponto de partida: a etapa de avaliação os ajusta.

## 009 — Duas estratégias de chunking: `fixed` e `structure`

- **O quê:** `fixed` = janelas deslizantes sobre o documento inteiro, ignorando a estrutura
  (linha de base; um trecho pode atravessar seções e recebe localização combinada).
  `structure` = um trecho por seção/página, dividindo só as grandes demais.
- **Alternativas:** chunking semântico (cortar onde a similaridade entre frases cai);
  chunking recursivo por separadores.
- **Por quê:** as duas são simples de explicar e formam uma comparação clara para as
  avaliações. O texto do trecho é um recorte do original, preservando quebras de linha e recuo
  de código. Na primeira ingestão do corpus, `structure` gerou 3.723 trechos (mediana de 82
  palavras, 352 com menos de 20) contra 2.145 do `fixed` (média de 195). Seções minúsculas
  como "Don't do this." perdem o sentido sem o título: incluir o caminho de títulos no texto
  indexado e juntar seções pequenas são experimentos candidatos para a etapa de avaliação.
