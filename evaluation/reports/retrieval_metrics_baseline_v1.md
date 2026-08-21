# Retrieval metrics — baseline_v1_retrieval

- **Embedding:** all-MiniLM-L6-v2 (384-dim) · **index:** 184 vectors · **dataset:** uniassist_eval_v1 v1.0.0-phase3
- **Scored:** 57 answerable questions (excluded 6 unanswerable/no-GT)
- **Reranking:** disabled for this run (dense-only baseline)
- **Index content:** RAW chunk text (LLM summaries unavailable — 402).

## Dense retrieval (top-20)

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | P@1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.877 | 0.965 | 0.983 | 0.983 | 0.912 | 0.877 | 0.632 | 0.517 | 0.390 |
| page | 0.632 | 0.807 | 0.877 | 0.965 | 0.708 | 0.632 | 0.242 | 0.142 | 0.082 |

> Source set-recall (distinct expected sources found): @1 0.877, @5 0.965, @10 0.983, @20 0.983

## Reranked (top-5)

| Level | Recall@1 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|
| source | 0.877 | 0.965 | 0.910 | 0.632 |
| page | 0.632 | 0.807 | 0.690 | 0.242 |

> Reranking was disabled (`--no-rerank`) for this dense-only baseline, so reranked rows equal the dense top-5. The Cohere reranker's true contribution is measured in Phase 9.

## By category (dense retrieval, source level)

| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 6 | 0.500 | 0.667 | 0.833 | 0.576 |
| direct_factual | 8 | 0.875 | 1.000 | 1.000 | 0.938 |
| exact_terminology | 7 | 0.714 | 1.000 | 1.000 | 0.833 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.857 | 1.000 | 1.000 | 0.886 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |

## By difficulty (dense retrieval, source level)

| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| easy | 15 | 0.667 | 0.867 | 0.933 | 0.764 |
| hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 37 | 0.946 | 1.000 | 1.000 | 0.960 |

> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.
