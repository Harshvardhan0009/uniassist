# Retrieval metrics — chunk_3500_400_e5_retrieval

- **Embedding:** intfloat/e5-base-v2 (768-dim) · **index:** 159 vectors · **dataset:** uniassist_eval_v1 v1.0.0-phase3
- **Scored:** 57 answerable questions (excluded 6 unanswerable/no-GT)
- **Reranking:** disabled for this run (dense-only baseline)
- **Index content:** RAW chunk text (LLM summaries unavailable — 402).

## Dense retrieval (top-20)

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | P@1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.860 | 0.983 | 1.000 | 1.000 | 0.917 | 0.860 | 0.656 | 0.523 | 0.390 |
| page | 0.649 | 0.842 | 0.895 | 0.947 | 0.741 | 0.649 | 0.228 | 0.121 | 0.067 |

> Source set-recall (distinct expected sources found): @1 0.860, @5 0.983, @10 1.000, @20 1.000

## Reranked (top-5)

| Level | Recall@1 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|
| source | 0.860 | 0.983 | 0.914 | 0.656 |
| page | 0.649 | 0.842 | 0.729 | 0.228 |

> Reranking was disabled (`--no-rerank`) for this dense-only baseline, so reranked rows equal the dense top-5. The Cohere reranker's true contribution is measured in Phase 9.

## By category (dense retrieval, source level)

| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 6 | 0.500 | 0.833 | 1.000 | 0.625 |
| direct_factual | 8 | 0.875 | 1.000 | 1.000 | 0.938 |
| exact_terminology | 7 | 0.714 | 1.000 | 1.000 | 0.857 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.714 | 1.000 | 1.000 | 0.857 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |

## By difficulty (dense retrieval, source level)

| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| easy | 15 | 0.733 | 1.000 | 1.000 | 0.856 |
| hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 37 | 0.892 | 0.973 | 1.000 | 0.930 |

> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.
