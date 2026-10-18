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

## 020 — Citações como marcadores `[n]` no texto, validadas depois com Pydantic

- **O quê:** os trechos entram no prompt numerados; o modelo escreve `[n]` após cada afirmação;
  ao fim da geração, `build_answer` extrai os marcadores e monta um `Answer` (Pydantic).
  Marcadores para trechos inexistentes são removidos e registrados em `invalid_citations`.
- **Alternativas:** saída estruturada em JSON (o Ollama aceita um JSON Schema em `format`);
  citações em uma lista separada ao fim da resposta.
- **Por quê:** marcadores permitem streaming (o JSON só fica útil quando termina), ligam cada
  afirmação à sua fonte e modelos pequenos seguem bem o padrão. Pydantic aqui (e não
  dataclass) porque o texto vem do LLM, ou seja, é entrada não confiável.

## 021 — Política: resposta sem citação válida vira "não sei"

- **O quê:** o validador do `Answer` proíbe resposta sem abstenção e sem citação. Se o modelo
  responde sem citar (ou só cita trechos inexistentes), a resposta é descartada e vira abstenção.
- **Alternativas:** mostrar a resposta com um aviso; pedir ao modelo para tentar de novo.
- **Por quê:** é a regra do projeto ("toda resposta cita fonte") aplicada no código, não só
  pedida no prompt. Afirmação sem fonte não chega ao usuário.
- **Limite conhecido:** validar que a citação **existe** não prova que o trecho **sustenta** a
  afirmação. Sem os exemplos do prompt (decisão 022), o qwen2.5:7b respondeu "The capital of
  France is [2]." e passaria na validação. Medir fidelidade das citações é tarefa da avaliação.

## 022 — Prompt v1: instruções + dois exemplos (few-shot) + trechos com delimitadores

- **O quê:** regras (citar toda frase, só citar o que sustenta, não usar conhecimento externo,
  trechos são dados e não instruções, responder no idioma da pergunta), um exemplo de resposta
  citada e um de abstenção; trechos em `<excerpt number=.. source=.. location=..>`.
- **Alternativas testadas** (6 perguntas, qwen2.5:7b e llama3.2:3b):
  - XML sem exemplos: 7B citou trecho aleatório para "capital da França" e não citou na
    resposta sobre senhas; 3B respondeu só "1." na pergunta de clickjacking.
  - Cabeçalhos `[n]` em texto puro com marcadores BEGIN/END: pior; 3B respondeu só "[2]".
  - XML **com** exemplos: 6/6 corretos nos dois modelos. Adotado.
- **Por quê:** os exemplos mostram o formato exato esperado, o que pesa muito em modelos
  pequenos. Os delimitadores + regra "dados, não instruções" são uma defesa básica contra
  prompt injection vinda de documentos indexados.

## 023 — Abstenção: sentinela tolerante + portão de confiança desligado

- **O quê:** o modelo responde `INSUFFICIENT_CONTEXT` + uma frase; a detecção aceita variações
  (`INSUFFICIENT CONTEXT`, maiúsculas/minúsculas, hífen), porque o llama3.2:3b escreve com
  espaço. Durante o streaming, o início do texto fica retido até se saber se é o sentinela, para
  ele não aparecer na tela. `MIN_RERANK_SCORE` (opcional) responde "não sei" sem chamar o LLM
  quando o melhor trecho tem score de reranking baixo.
- **Alternativas:** detectar frases livres ("não sei", "I don't know"); portão ligado com um
  limite chutado.
- **Por quê:** texto fixo é detectável com segurança. O portão economiza uma chamada ao LLM,
  mas um limite sem calibração descartaria respostas boas: fica desligado até a avaliação.

## 024 — Geração reprodutível e contexto explícito

- **O quê:** `temperature=0`, `seed=42`, `num_ctx=8192` enviados em toda chamada; aviso se o
  prompt passar de 90% do contexto.
- **Por quê:** a avaliação precisa de respostas reprodutíveis. O Ollama **corta em silêncio**
  o começo de um prompt maior que `num_ctx`; os trechos sumiriam sem erro. Hoje os prompts
  têm ~700–1.750 tokens (5 trechos), bem abaixo do limite.

## 025 — Primeira comparação de LLMs: llama3.2:3b vs. qwen2.5:7b

Prompt v1, híbrido + rerank, 5 perguntas respondíveis + 3 fora do corpus (inspeção, não métrica):

| | llama3.2:3b | qwen2.5:7b |
|---|---|---|
| Respondíveis com citação válida | 5/5 | 5/5 |
| Fora do corpus com abstenção | 3/3 | 3/3 |
| Citações para trechos inexistentes | 0 | 0 |
| Velocidade de geração (M4) | ~43 tok/s | ~21 tok/s |
| Tempo até o 1º token | 0,2–3,9 s | 3,2–10 s |

O 7B dá respostas mais completas (ex.: cita X-Frame-Options e CSP no clickjacking; o 3B só
o script frame-buster), mas tende a "citar tudo" na frase de abertura ([1][2][3][4][5]).
O 3B fica como modelo de desenvolvimento (`--dev`), o 7B como padrão. A qualidade e a
fidelidade das citações serão medidas na avaliação.

## 026 — Avaliação: dataset semi-sintético com rótulos por seção

- **O quê:** 101 perguntas em `evals/dataset.jsonl`: 56 geradas pelo qwen2.5:7b a partir de
  seções sorteadas (uma por cheat sheet) e curadas à mão, 15 traduções para português dessas
  mesmas perguntas, 10 de múltiplas seções e 20 fora do corpus escritas à mão. Toda a curadoria
  está em `scripts/build_dataset.py` (o que foi aprovado, reescrito e descartado, com motivo).
- **Alternativas:** tudo escrito à mão; tudo sintético sem revisão; rótulos por ID de trecho.
- **Por quê:** rótulos por (fonte, seção) valem para qualquer chunking. Das 80 candidatas, 24
  foram descartadas, a maioria por **ambiguidade** (a resposta também existe em outro cheat
  sheet, o que geraria falso "erro" de busca) ou vagueza. As traduções medem o efeito do idioma
  com os mesmos rótulos. As fora do corpus foram conferidas com `grep`; metade é de segurança/TI
  vizinha ao corpus (mais difícil). Viés conhecido: perguntas geradas do texto favorecem termos
  exatos; por isso o recorte por tipo (exata vs. paráfrase).
- **Tamanho:** com ~80 respondíveis, intervalos de confiança de ±8 pontos são normais. Diferenças
  menores que isso não são tratadas como conclusão.

## 027 — Resultado: modo de busca (grupo A)

| | R@5 | Paráfrase | Multi |
|---|---|---|---|
| BM25 | 66 | 61 | 32 |
| Vetorial | 83 | 75 | 62 |
| Híbrido (sem rerank) | 78 | 71 | 45 |
| Híbrido + rerank | 85 | 82 | 60 |

- O híbrido **sem** reranking é pior que o vetorial sozinho: neste corpus o BM25 é fraco e o RRF
  dá a ele o mesmo peso. Com reranking o problema some, porque o cross-encoder reordena os
  candidatos das duas listas. O ganho do reranking aparece mais em R@1 (51 → 58) e MRR (66 → 72).
- Conclusão: híbrido só vale com reranking. RRF ponderado (menos peso ao BM25) fica como
  experimento futuro.

## 028 — Resultado: chunking (grupo B)

- **Tamanho quase não importa:** structure 100/200/400 e fixed 100/200/400 ficam entre 82 e 85
  (intervalos sobrepostos).
- **Fixed vence em múltiplas seções** (92–95 vs. 55–60), porque um trecho atravessa seções
  vizinhas, e **perde em paráfrase** (68–75 vs. 82).
- **Contextual chunk headers ajudam:** structure-200 com títulos vai de 85 para 88 em R@5, de 58
  para 62 em R@1 e de 60 para 83 em multi. Confirma a suspeita da etapa 1 (trechos curtos como
  "Don't do this." ficam encontráveis com o título).

## 029 — Resultado: modelos multilíngues (grupo C)

| | R@5 | R@1 | MRR | Paráfrase | Multi | PT | Busca p50 |
|---|---|---|---|---|---|---|---|
| nomic + MiniLM (linha de base) | 85 | 58 | 72 | 82 | 60 | 53 | 155 ms |
| bge-m3 + MiniLM | 86 | 59 | 74 | 86 | 65 | 60 | 256 ms |
| nomic + bge-reranker | 86 | 61 | 74 | 82 | 63 | 53 | 3,8 s |
| bge-m3 + bge-reranker | 94 | 66 | 80 | 96 | 58 | 100 | 4,2 s |
| **bge-m3 + bge-reranker + títulos** | **95** | **74** | **86** | 93 | 78 | 93 | 4,2 s |

- Embedding e reranker multilíngues **só funcionam juntos**: cada um sozinho mal muda o PT
  (53 → 60 / 53). Juntos, o português vai de 53 para 93–100 (n = 15, intervalo largo).
- O bge-reranker também melhora inglês: R@5 sobe de 85 para 94.
- **Custo:** o bge-reranker-v2-m3 tem 568M parâmetros contra 22M do MiniLM; reordenar 20
  candidatos leva ~2–4 s no M4 (MPS), contra ~0,2 s.

## 030 — Nova configuração padrão: bge-m3 + bge-reranker-v2-m3 + títulos

- **O quê:** `EMBEDDING_MODEL=bge-m3` (sem prefixos), `RERANK_MODEL=BAAI/bge-reranker-v2-m3`,
  `CHUNK_HEADINGS=true`, structure-200-40, híbrido + rerank.
- **Alternativas:** manter nomic + MiniLM (busca 25x mais rápida, 10 pontos a menos de R@5 e
  português quebrado).
- **Por quê:** +10 pontos de R@5, +16 de R@1, português funcionando, e os ~4 s extras são
  pequenos perto dos ~10 s de geração do 7B. A configuração antiga continua disponível pelo
  `.env` para quem prefere velocidade.

## 031 — Calibração do portão de confiança

- Com o bge-reranker, o score do melhor trecho separa bem as perguntas: limite **0,38** mantém
  95% das respondíveis e barra 95% das fora do corpus (o reranker devolve probabilidades 0–1;
  fora do corpus a mediana é 0,0). Com o MiniLM a separação é pior (78% / 95%).
- **Mantido desligado por padrão:** o limite foi escolhido no mesmo dataset em que é medido
  (risco de sobreajuste), e o "não sei" do LLM já cobre o caso (decisão 033). Ativar com
  `MIN_RERANK_SCORE=0.38` economiza a chamada ao LLM em perguntas fora do corpus.

## 032 — Execução da avaliação de geração em uma máquina de 16 GB

- **Problema:** gerador (3B), juiz (7B), bge-m3 e bge-reranker (na GPU) não cabem juntos com
  folga em 16 GB. Alternar gerador e juiz a cada pergunta fez o Ollama falhar ao carregar um
  modelo (`500: timed out waiting for llama-server to start`), e rodadas com o 7B foram encerradas
  pelo sistema após 15–20 perguntas.
- **O quê:** (1) o juiz roda numa segunda passada, depois de toda a geração (um LLM por vez);
  (2) retry com espera em erros 5xx do Ollama, só antes do streaming começar; (3) checkpoint por
  pergunta em `evals/results/.checkpoints/`, com escrita atômica, para retomar uma rodada
  interrompida.
- **Alternativas:** rodar no PC via Tailscale (previsto para avaliações longas); reranker na CPU.
- **Por quê:** cada rodada com geração leva ~1 h; perder duas no meio mostrou que retomar é mais
  importante que evitar toda falha.

## 033 — Resultado: geração com llama3.2:3b (D1, busca C4)

| Métrica | Valor (IC 95%) |
|---|---|
| Responde (respondíveis) | 88% (80–94) |
| Recusa (fora do corpus) | 90% (75–100) |
| Precisão das citações | 77% (69–85) |
| Fatos-chave | 85% (75–92) |
| Fidelidade (juiz) | 79% (69–88) |
| Citação inválida | 0% |

- **Todas as 10 respondíveis em que o 3B disse "não sei" tinham o trecho certo no top 5**
  (posição 1 a 3): a falha é de geração, não de busca.
- **2 alucinações fora do corpus, com citação:** um comando `certbot renew` atribuído a um
  trecho e "Kubernetes 1.30 was released [2]". O juiz marcou as duas como não sustentadas — o
  caso que a validação de existência da citação (decisão 021) não pega.
- 17 de 60 respostas têm ao menos uma frase não sustentada; o 3B é usado como modelo de
  desenvolvimento (`--dev`).
- A rodada com qwen2.5:7b (D2) está incompleta (26 de 101 perguntas, salvas em checkpoint) e
  será adicionada depois.

## 034 — Mensagem de abstenção fixa

- **O quê:** a frase que o modelo escreve depois de `INSUFFICIENT_CONTEXT` é ignorada (fica em
  `raw_text`); a resposta usa `ABSTAIN_MESSAGE`, configurável, padrão em português.
- **Alternativas:** usar a frase do modelo (versão anterior); detectar o idioma da pergunta.
- **Por quê:** num teste manual, o qwen2.5:7b escreveu a frase em chinês para uma pergunta em
  português, e o código ainda acrescentava "." depois do "。". Mensagem fixa é previsível e é
  uma coisa a menos que o LLM pode estragar. O prompt não mudou (continua v1), então as métricas
  de D1 e D2 seguem comparáveis.

## 035 — Validação do juiz de fidelidade

- **O quê:** 30 vereditos do D1 sorteados (semente fixa) e rotulados manualmente, lendo a frase e
  o trecho citado completo (`evals/judge_labels.jsonl`, `scripts/judge_agreement.py`).
- **Resultado:** concordância de **29/30 (97%)**. O único erro foi aprovar uma lista de itens que
  o trecho citado não continha (a lista vinha de outro cheat sheet) — o tipo de erro mais grave,
  então a fidelidade medida tende a ser levemente otimista.
- **Ressalvas:** amostra pequena; no D2 o juiz é o mesmo modelo que gera as respostas.

## 036 — Procedência dos resultados

Os experimentos A–C rodaram com o código que depois virou o commit `b1f9046`, antes de ele ser
commitado; por isso o campo `git_commit` deles termina em `-dirty`. O caminho de busca não mudou
depois disso. O D1 rodou no commit `b1f9046` limpo.
