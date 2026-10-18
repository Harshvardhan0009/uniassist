# Retrieval metrics — hybrid_e5_bm25_1500_retrieval

- **Embedding:** intfloat/e5-base-v2 (768-dim) · **index:** 285 vectors · **dataset:** uniassist_eval_v1 v1.0.0-phase3
- **Scored:** 57 answerable questions (excluded 6 unanswerable/no-GT)
- **Reranking:** disabled for this run (dense-only baseline)
- **Index content:** RAW chunk text (LLM summaries unavailable — 402).

## Dense retrieval (top-20)

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | P@1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.965 | 1.000 | 1.000 | 1.000 | 0.983 | 0.965 | 0.807 | 0.667 | 0.527 |
| page | 0.825 | 0.895 | 0.930 | 0.983 | 0.863 | 0.825 | 0.383 | 0.225 | 0.131 |

> Source set-recall (distinct expected sources found): @1 0.965, @5 1.000, @10 1.000, @20 1.000

## Reranked (top-5)

| Level | Recall@1 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|
| source | 0.965 | 1.000 | 0.983 | 0.807 |
| page | 0.825 | 0.895 | 0.854 | 0.383 |

> Reranking was disabled (`--no-rerank`) for this dense-only baseline, so reranked rows equal the dense top-5. The Cohere reranker's true contribution is measured in Phase 9.

## By category (dense retrieval, source level)

| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 6 | 0.833 | 1.000 | 1.000 | 0.917 |
| direct_factual | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_terminology | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.857 | 1.000 | 1.000 | 0.929 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |

## By difficulty (dense retrieval, source level)

| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| easy | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 37 | 0.946 | 1.000 | 1.000 | 0.973 |

> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.
