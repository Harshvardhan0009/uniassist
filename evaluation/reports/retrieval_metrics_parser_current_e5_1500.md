# Retrieval metrics — parser_current_e5_1500_retrieval

- **Embedding:** intfloat/e5-base-v2 (768-dim) · **index:** 285 vectors · **dataset:** uniassist_eval_v1 v1.0.0-phase3
- **Scored:** 57 answerable questions (excluded 6 unanswerable/no-GT)
- **Reranking:** applied on 63/63 questions
- **Index content:** RAW chunk text (LLM summaries unavailable — 402).

## Dense retrieval (top-20)

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | P@1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.930 | 1.000 | 1.000 | 1.000 | 0.952 | 0.930 | 0.775 | 0.675 | 0.517 |
| page | 0.789 | 0.860 | 0.930 | 0.965 | 0.819 | 0.789 | 0.368 | 0.226 | 0.128 |

> Source set-recall (distinct expected sources found): @1 0.930, @5 1.000, @10 1.000, @20 1.000

## Reranked (top-5)

| Level | Recall@1 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|
| source | 0.930 | 1.000 | 0.953 | 0.796 |
| page | 0.877 | 0.965 | 0.906 | 0.393 |

## By category (dense retrieval, source level)

| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 6 | 0.667 | 1.000 | 1.000 | 0.742 |
| direct_factual | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_terminology | 7 | 0.857 | 1.000 | 1.000 | 0.905 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.857 | 1.000 | 1.000 | 0.929 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |

## By difficulty (dense retrieval, source level)

| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| easy | 15 | 0.867 | 1.000 | 1.000 | 0.906 |
| hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 37 | 0.946 | 1.000 | 1.000 | 0.965 |

> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.
