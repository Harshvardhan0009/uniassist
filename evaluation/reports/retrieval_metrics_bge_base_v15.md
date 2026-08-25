# Retrieval metrics — bge_base_v15_retrieval

- **Embedding:** BAAI/bge-base-en-v1.5 (768-dim) · **index:** 184 vectors · **dataset:** uniassist_eval_v1 v1.0.0-phase3
- **Scored:** 57 answerable questions (excluded 6 unanswerable/no-GT)
- **Reranking:** disabled for this run (dense-only baseline)
- **Index content:** RAW chunk text (LLM summaries unavailable — 402).

## Dense retrieval (top-20)

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | P@1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.877 | 0.983 | 0.983 | 0.983 | 0.915 | 0.877 | 0.674 | 0.540 | 0.425 |
| page | 0.614 | 0.825 | 0.895 | 0.965 | 0.713 | 0.614 | 0.256 | 0.146 | 0.083 |

> Source set-recall (distinct expected sources found): @1 0.877, @5 0.983, @10 0.983, @20 0.983

## Reranked (top-5)

| Level | Recall@1 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|
| source | 0.877 | 0.983 | 0.915 | 0.674 |
| page | 0.614 | 0.825 | 0.698 | 0.256 |

> Reranking was disabled (`--no-rerank`) for this dense-only baseline, so reranked rows equal the dense top-5. The Cohere reranker's true contribution is measured in Phase 9.

## By category (dense retrieval, source level)

| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 6 | 0.333 | 1.000 | 1.000 | 0.569 |
| direct_factual | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_terminology | 7 | 0.714 | 0.857 | 0.857 | 0.786 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.857 | 1.000 | 1.000 | 0.893 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |

## By difficulty (dense retrieval, source level)

| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| easy | 15 | 0.733 | 0.933 | 0.933 | 0.822 |
| hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 37 | 0.919 | 1.000 | 1.000 | 0.941 |

> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.
