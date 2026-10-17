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

## 010 — Embeddings calculados por nós, via API HTTP do Ollama

- **O quê:** `OllamaEmbedder` chama `POST /api/embed` com `httpx`, em lotes de 32 textos, e
  entrega os vetores prontos ao Chroma (`embedding_function=None`).
- **Alternativas:** função de embedding do próprio Chroma; SDK `ollama` para Python.
- **Por quê:** fica explícito o que é enviado e recebido; o modelo é sempre o do Ollama (o
  Chroma, por padrão, baixaria um modelo ONNX próprio); a interface `Embedder` permite um
  embedder falso nos testes, e o CI nunca precisa de modelo. Erros de conexão e de modelo
  ausente viram mensagens que dizem o que fazer.

## 011 — Prefixos de tarefa do `nomic-embed-text`

- **O quê:** trechos indexados com `search_document: ` e consultas com `search_query: `
  (configuráveis: `EMBED_DOC_PREFIX`, `EMBED_QUERY_PREFIX`).
- **Alternativas:** embeddar o texto puro.
- **Por quê:** o modelo foi treinado com esses prefixos; sem eles a qualidade da busca cai.
  Ficam na config porque outro modelo de embedding pode usar outros prefixos ou nenhum.

## 012 — Chroma: cosseno, uma coleção por configuração, sincronização por ID

- **O quê:** distância de cosseno (score = `1 − distância`); coleção nomeada pela configuração
  (ex.: `structure-200-40-nomic-embed-text`); `sync` calcula embeddings só dos IDs novos e
  remove os que sumiram.
- **Alternativas:** distância L2; uma coleção única recriada a cada ingestão.
- **Por quê:** cosseno é a métrica usual para embeddings de texto. Coleções separadas permitem
  comparar configurações lado a lado na avaliação. Graças aos IDs determinísticos, reindexar o
  corpus inteiro levou 97 s na primeira vez e 0,2 s na segunda (M4, 3.723 trechos).

## 013 — BM25 em memória, com tokenizador Unicode simples

- **O quê:** `rank-bm25` (BM25Okapi, k1=1,5, b=0,75) montado a partir do JSONL de trechos a
  cada busca; tokens = sequências de letras/dígitos Unicode em minúsculas.
- **Alternativas:** persistir o índice; tokenizador com stopwords/stemming; Elasticsearch.
- **Por quê:** montar o índice leva menos de 1 s para alguns milhares de trechos. O primeiro
  tokenizador (`[a-z0-9]+`) quebrava "saída" em "sa" + "da"; um teste revelou isso e ele passou
  a usar `[^\W_]+`. Stopwords e stemming ficam como experimentos da avaliação.
- **Cuidado conhecido:** no BM25Okapi, um termo presente em metade ou mais dos trechos recebe
  IDF ~0. Em corpus real isso só afeta palavras muito comuns; em testes com 2 trechos, zera tudo.

## 014 — Primeira inspeção qualitativa: vetorial vs. BM25

Cinco consultas, top 3, estratégia `structure-200-40` (não é métrica, só inspeção):

- **Termo exato** (`X-Frame-Options`, `JWT alg none`): os dois acertam; o vetorial pôs a seção
  mais específica em primeiro.
- **Paráfrase** ("how should I store user passwords safely"): o vetorial acertou
  (Password Storage › Introduction); o BM25 pôs uma seção de MFA em primeiro.
- **Paráfrase difícil** ("stop other websites from loading my page inside an iframe"): nenhum
  dos dois trouxe a defesa correta (X-Frame-Options / `frame-ancestors`) no top 3.
- **Pergunta em português**: os dois falham. O `nomic-embed-text` é treinado em inglês, e o BM25
  casou "com" com domínios ".com". Se o material do mestrado estiver em português, a avaliação
  precisa comparar um modelo de embedding multilíngue.
- **Escalas:** cossenos ficam entre ~0,5 e 0,8; scores BM25 entre ~6 e 24. Não são comparáveis,
  o que motiva fundir por **posição** (RRF) e não por score na etapa 3.

## 015 — Fusão por Reciprocal Rank Fusion (RRF), k = 60

- **O quê:** `score(d) = Σ 1/(k + posição_i(d))` sobre os buscadores em que o trecho aparece;
  `k = 60`. Cada resultado guarda de onde veio (`vetorial #3 · bm25 #1`).
- **Alternativas:** soma ponderada de scores normalizados (min-max, z-score); RRF ponderado.
- **Por quê:** os scores vivem em escalas diferentes (decisão 014) e normalizá-los é frágil
  (depende de quais resultados vieram na lista). O RRF só usa posições, não tem parâmetro a
  ajustar além de `k`, e `k = 60` é o valor do artigo original (Cormack et al., 2009).
  Pesos por buscador ficam como experimento para a avaliação.

## 016 — Funil de candidatos: 20 por buscador → reranking → top 5

- **O quê:** cada buscador entrega `CANDIDATES = 20`; o RRF funde; o cross-encoder reordena os 20
  e devolve `TOP_K = 5`.
- **Alternativas:** reranking direto sobre o top 5 (barato, mas não recupera o que ficou fora);
  candidatos = 50 ou 100 (mais recall, mais latência).
- **Por quê:** o reranking só consegue promover o que está entre os candidatos. Na pergunta do
  iframe (decisão 019), as respostas certas estavam em 13º e 16º antes do reranking; com
  candidatos = 5 elas nunca apareceriam. 20 é ponto de partida; a avaliação mede recall@N.

## 017 — Cross-encoder inicial: `cross-encoder/ms-marco-MiniLM-L-6-v2`

- **O quê:** cross-encoder via `sentence-transformers` (configurável em `RERANK_MODEL`),
  carregado só quando o reranking está ligado (import tardio).
- **Alternativas:** `BAAI/bge-reranker-v2-m3` (multilíngue, ~2 GB); reranking com o LLM.
- **Por quê:** o corpus é em inglês; o MiniLM tem ~90 MB e reordena 20 candidatos em ~150–200 ms
  no M4 (MPS). O `bge-reranker-v2-m3` é o candidato da avaliação para perguntas em português,
  que continuam falhando (o reranker também é treinado em inglês).

## 018 — PyTorch CPU-only no Linux (CI)

- **O quê:** `torch` declarado como dependência direta, com `[tool.uv.sources]` apontando para o
  índice `https://download.pytorch.org/whl/cpu` só quando `sys_platform == 'linux'`.
- **Alternativas:** PyTorch padrão em todo lugar (no Linux traz CUDA, vários GB);
  reranking como dependência opcional fora do CI.
- **Por quê:** o CI instala exatamente o que o projeto usa, sem GBs de CUDA inútil. No macOS
  continua o pacote do PyPI, que usa a GPU do Mac. Detalhe aprendido: o uv só aplica `sources`
  a dependências **diretas**, por isso `torch` precisou entrar no `pyproject.toml`.

## 019 — Inspeção qualitativa: vetorial, BM25, híbrido e híbrido + rerank

As cinco consultas da decisão 014, top 3, `structure-200-40` (inspeção, não métrica):

- **Termo exato** (`X-Frame-Options`, `JWT alg none`): todas as configurações acertam; o
  reranking pôs em 1º a seção que define o header.
- **Paráfrase** (senhas): híbrido corrigiu o erro do BM25 (Password Storage em 1º e 2º).
- **Paráfrase difícil** (iframe): só o **híbrido + rerank** acertou: Clickjacking › Introduction
  e Defending with X-Frame-Options subiram de 13º e 16º para 1º e 3º. Melhor argumento a favor
  do funil de 20 candidatos.
- **Português**: continua falhando em todas (embedding e reranker em inglês).
- **Latência** (M4, por consulta): BM25 ~3 ms, vetorial ~20 ms, híbrido ~20 ms,
  híbrido + rerank ~170–225 ms. Carregar o cross-encoder custa alguns segundos, uma vez por processo.
