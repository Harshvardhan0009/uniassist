# Retrieval metrics — parser_unstructured_e5_1500_retrieval

- **Embedding:** intfloat/e5-base-v2 (768-dim) · **index:** 231 vectors · **dataset:** uniassist_eval_v1 v1.0.0-phase3
- **Scored:** 57 answerable questions (excluded 6 unanswerable/no-GT)
- **Reranking:** applied on 63/63 questions
- **Index content:** RAW chunk text (LLM summaries unavailable — 402).

## Dense retrieval (top-20)

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | P@1 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|---|---|---|---|
| source | 0.895 | 1.000 | 1.000 | 1.000 | 0.932 | 0.895 | 0.709 | 0.560 | 0.450 |
| page | 0.737 | 0.877 | 0.895 | 0.965 | 0.792 | 0.737 | 0.305 | 0.172 | 0.098 |

> Source set-recall (distinct expected sources found): @1 0.895, @5 1.000, @10 1.000, @20 1.000

## Reranked (top-5)

| Level | Recall@1 | Recall@5 | MRR | P@5 |
|---|---|---|---|---|
| source | 0.965 | 1.000 | 0.977 | 0.747 |
| page | 0.877 | 0.965 | 0.912 | 0.319 |

## By category (dense retrieval, source level)

| Category | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| conversational | 6 | 0.500 | 1.000 | 1.000 | 0.631 |
| direct_factual | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_terminology | 7 | 0.714 | 1.000 | 1.000 | 0.833 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.857 | 1.000 | 1.000 | 0.929 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |

## By difficulty (dense retrieval, source level)

| Difficulty | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| easy | 15 | 0.800 | 1.000 | 1.000 | 0.869 |
| hard | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| medium | 37 | 0.919 | 1.000 | 1.000 | 0.948 |

> Metric definitions: see `evaluation/metrics/retrieval_metrics.py` module docstring.
