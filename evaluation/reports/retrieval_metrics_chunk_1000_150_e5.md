# Retrieval metrics — chunk_1000_150_e5_retrieval

- **Embedding:** intfloat/e5-base-v2 (768-dim) · **index:** 384 vectors · **dataset:** uniassist_eval_v1 v1.0.0-phase3
- **Scored:** 57 answerable questions (excluded 6 unanswerable/no-GT)
- **Reranking:** disabled for this run (dense-only baseline)
- **Index content:** RAW chunk text (LLM summaries unavailable — 402).

## Dense retrieval (top-20)

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | P@1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.912 | 1.000 | 1.000 | 1.000 | 0.939 | 0.912 | 0.800 | 0.693 | 0.565 |
| page | 0.702 | 0.860 | 0.912 | 0.983 | 0.776 | 0.702 | 0.421 | 0.260 | 0.156 |

> Source set-recall (distinct expected sources found): @1 0.912, @5 1.000, @10 1.000, @20 1.000

## Reranked (top-5)

| Level | Recall@1 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|
| source | 0.912 | 1.000 | 0.939 | 0.800 |
| page | 0.702 | 0.860 | 0.765 | 0.421 |

> Reranking was disabled (`--no-rerank`) for this dense-only baseline, so reranked rows equal the dense top-5. The Cohere reranker's true contribution is measured in Phase 9.

## By category (dense retrieval, source level)

| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 6 | 0.667 | 1.000 | 1.000 | 0.742 |
| direct_factual | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_terminology | 7 | 0.857 | 1.000 | 1.000 | 0.929 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.714 | 1.000 | 1.000 | 0.798 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |

## By difficulty (dense retrieval, source level)

| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| easy | 15 | 0.867 | 1.000 | 1.000 | 0.917 |
| hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 37 | 0.919 | 1.000 | 1.000 | 0.940 |

> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.
