# Resultados da avaliação

Gerado por `grounded-rag eval --report-only`. Não edite à mão.

- Perguntas: 101 · corpus OWASP @ `1a78ae3` · prompt v1
- Valores em %, com intervalo de confiança de 95% por bootstrap entre parênteses.
- R@k = recall@k (seções relevantes no top k); recortes por tipo usam R@5.

## Busca

| Experimento | Índice | Busca | R@1 | R@5 | R@20 | MRR@10 | R@5 exato | R@5 paráfrase | R@5 multi | R@5 PT | Busca p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C4-bge-m3-bge-reranker-headings | structure-200-40-h · bge-m3 | hybrid + rerank | 74 (64–83) | **95 (90–99)** | 98 (94–100) | 86 (80–91) | 100 (100–100) | 93 (82–100) | 78 (60–95) | 93 (80–100) | 4.247 |
| D1-llama3b | structure-200-40-h · bge-m3 | hybrid + rerank | 74 (64–83) | **95 (90–99)** | 98 (94–100) | 86 (80–91) | 100 (100–100) | 93 (82–100) | 78 (60–95) | 93 (80–100) | 10.852 |
| C3-bge-m3-bge-reranker | structure-200-40 · bge-m3 | hybrid + rerank | 66 (56–76) | **94 (89–98)** | 98 (94–100) | 80 (74–87) | 100 (100–100) | 96 (89–100) | 58 (40–78) | 100 (100–100) | 4.239 |
| B6-structure-200-headings | structure-200-40-h · nomic-embed-text | hybrid + rerank | 62 (52–72) | **88 (80–94)** | 90 (82–95) | 77 (69–85) | 93 (84–100) | 82 (68–96) | 83 (70–95) | 53 (27–80) | 250 |
| C1-bge-m3 | structure-200-40 · bge-m3 | hybrid + rerank | 59 (49–69) | **86 (78–93)** | 98 (94–100) | 74 (66–81) | 91 (81–98) | 86 (71–96) | 65 (47–82) | 60 (33–80) | 256 |
| C2-bge-reranker | structure-200-40 · nomic-embed-text | hybrid + rerank | 61 (51–71) | **86 (78–92)** | 88 (81–94) | 74 (66–82) | 93 (84–100) | 82 (68–96) | 63 (48–80) | 53 (27–80) | 3.845 |
| B7-fixed-200-headings | fixed-200-40-h · nomic-embed-text | hybrid + rerank | 67 (57–77) | **85 (77–93)** | 94 (88–99) | 76 (68–84) | 91 (81–98) | 75 (57–89) | 92 (80–100) | 60 (33–87) | 275 |
| A6-hybrid-rerank | structure-200-40 · nomic-embed-text | hybrid + rerank | 58 (47–68) | **85 (77–92)** | 88 (81–94) | 72 (64–80) | 93 (84–100) | 82 (68–96) | 60 (37–80) | 53 (27–80) | 155 |
| B2-structure-100 | structure-100-20 · nomic-embed-text | hybrid + rerank | 59 (48–69) | **83 (75–91)** | 88 (80–94) | 72 (64–80) | 91 (81–98) | 82 (68–96) | 55 (35–73) | 53 (27–80) | 156 |
| B3-structure-400 | structure-400-80 · nomic-embed-text | hybrid + rerank | 54 (44–65) | **83 (75–90)** | 88 (81–95) | 70 (62–78) | 91 (81–98) | 82 (68–96) | 55 (35–73) | 47 (20–73) | 256 |
| B5-fixed-400 | fixed-400-80 · nomic-embed-text | hybrid + rerank | 58 (47–69) | **83 (75–91)** | 95 (90–99) | 71 (62–79) | 91 (81–98) | 68 (50–86) | 95 (85–100) | 60 (33–87) | 278 |
| B8-vector-headings | structure-200-40-h · nomic-embed-text | vector | 64 (53–73) | **83 (75–90)** | 93 (86–98) | 76 (68–84) | 93 (84–100) | 68 (50–86) | 83 (70–97) | 47 (20–73) | 18 |
| A1-vector | structure-200-40 · nomic-embed-text | vector | 53 (42–63) | **83 (75–90)** | 89 (81–95) | 68 (59–75) | 93 (84–100) | 75 (57–89) | 62 (43–82) | 53 (27–80) | 29 |
| B1-fixed-200 | fixed-200-40 · nomic-embed-text | hybrid + rerank | 61 (50–71) | **83 (74–90)** | 93 (86–98) | 72 (62–80) | 91 (81–98) | 68 (50–86) | 92 (80–100) | 60 (33–87) | 261 |
| A4-vector-rerank | structure-200-40 · nomic-embed-text | vector + rerank | 59 (49–69) | **83 (74–90)** | 89 (81–95) | 73 (65–81) | 88 (77–98) | 82 (68–96) | 60 (37–80) | 47 (20–73) | 150 |
| B4-fixed-100 | fixed-100-20 · nomic-embed-text | hybrid + rerank | 57 (46–67) | **82 (74–90)** | 91 (85–96) | 71 (62–78) | 86 (74–95) | 79 (61–93) | 77 (60–92) | 53 (27–80) | 157 |
| A3-hybrid | structure-200-40 · nomic-embed-text | hybrid | 51 (40–62) | **78 (69–87)** | 88 (81–94) | 66 (57–74) | 91 (81–98) | 71 (54–86) | 45 (20–72) | 47 (20–73) | 34 |
| A5-bm25-rerank | structure-200-40 · nomic-embed-text | bm25 + rerank | 54 (44–65) | **77 (67–85)** | 79 (69–87) | 67 (57–76) | 86 (74–95) | 68 (50–86) | 60 (37–80) | 40 (20–67) | 138 |
| A2-bm25 | structure-200-40 · nomic-embed-text | bm25 | 42 (31–52) | **66 (55–75)** | 79 (69–87) | 55 (45–64) | 77 (65–88) | 61 (43–79) | 32 (13–52) | 40 (20–67) | 4 |

## Resposta

| Experimento | LLM | Responde (respondíveis) | Recusa (fora do corpus) | Precisão das citações | Fatos-chave | Fidelidade | Citação inválida | 1º token p50 (ms) | Total p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| D1-llama3b | llama3.2:3b | 88 (80–94) | 90 (75–100) | 77 (69–85) | 85 (75–92) | 79 (69–88) | 0 (0–0) | 10.922 | 14.904 / 24.576 |

## Calibração do portão de confiança (MIN_RERANK_SCORE)

Score do melhor trecho após reranking; o limite maximiza a acurácia balanceada.

| Experimento | Limite | Respondíveis mantidas | Fora do corpus barradas | Score respondíveis (mín / mediana) | Score fora do corpus (máx / mediana) |
|---|---|---|---|---|---|
| A4-vector-rerank | 0.56 | 78% | 95% | -11.4 / 4.8 | 2.8 / -5.3 |
| A5-bm25-rerank | 0.77 | 77% | 95% | -11.4 / 4.8 | 2.8 / -6.4 |
| A6-hybrid-rerank | 0.56 | 78% | 95% | -11.4 / 4.8 | 2.8 / -5.2 |
| B1-fixed-200 | 1.39 | 74% | 100% | -11.4 / 4.1 | 1.4 / -7.2 |
| B2-structure-100 | 1.53 | 77% | 95% | -11.4 / 4.7 | 4.3 / -5.2 |
| B3-structure-400 | 0.56 | 78% | 95% | -11.4 / 4.8 | 2.7 / -6.4 |
| B4-fixed-100 | 1.50 | 77% | 95% | -11.5 / 4.7 | 2.9 / -5.1 |
| B5-fixed-400 | 0.40 | 79% | 100% | -11.3 / 3.6 | -0.1 / -7.1 |
| B6-structure-200-headings | -0.57 | 83% | 95% | -11.4 / 5.3 | 2.1 / -8.4 |
| B7-fixed-200-headings | -0.93 | 80% | 100% | -11.4 / 3.9 | -1.0 / -7.9 |
| C1-bge-m3 | 0.56 | 78% | 95% | -11.4 / 4.8 | 2.8 / -5.2 |
| C2-bge-reranker | 0.50 | 91% | 100% | 0.0 / 0.9 | 0.4 / 0.0 |
| C3-bge-m3-bge-reranker | 0.50 | 95% | 100% | 0.0 / 1.0 | 0.4 / 0.0 |
| C4-bge-m3-bge-reranker-headings | 0.38 | 95% | 95% | 0.0 / 1.0 | 0.6 / 0.0 |
| D1-llama3b | 0.38 | 95% | 95% | 0.0 / 1.0 | 0.6 / 0.0 |
