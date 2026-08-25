# Retrieval metrics — e5_base_v2_retrieval

- **Embedding:** intfloat/e5-base-v2 (768-dim) · **index:** 184 vectors · **dataset:** uniassist_eval_v1 v1.0.0-phase3
- **Scored:** 57 answerable questions (excluded 6 unanswerable/no-GT)
- **Reranking:** disabled for this run (dense-only baseline)
- **Index content:** RAW chunk text (LLM summaries unavailable — 402).

## Dense retrieval (top-20)

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | P@1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.877 | 1.000 | 1.000 | 1.000 | 0.927 | 0.877 | 0.709 | 0.565 | 0.427 |
| page | 0.649 | 0.842 | 0.930 | 0.965 | 0.743 | 0.649 | 0.281 | 0.154 | 0.084 |

> Source set-recall (distinct expected sources found): @1 0.877, @5 1.000, @10 1.000, @20 1.000

## Reranked (top-5)

| Level | Recall@1 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|
| source | 0.877 | 1.000 | 0.927 | 0.709 |
| page | 0.649 | 0.842 | 0.727 | 0.281 |

> Reranking was disabled (`--no-rerank`) for this dense-only baseline, so reranked rows equal the dense top-5. The Cohere reranker's true contribution is measured in Phase 9.

## By category (dense retrieval, source level)

| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 6 | 0.500 | 1.000 | 1.000 | 0.681 |
| direct_factual | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_terminology | 7 | 0.714 | 1.000 | 1.000 | 0.857 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.714 | 1.000 | 1.000 | 0.821 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |

## By difficulty (dense retrieval, source level)

| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| easy | 15 | 0.733 | 1.000 | 1.000 | 0.856 |
| hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 37 | 0.919 | 1.000 | 1.000 | 0.946 |

> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.
