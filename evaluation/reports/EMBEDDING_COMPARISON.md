# Phase 7 — Embedding model comparison

Dense retrieval on the **frozen `baseline_v1` snapshot** (raw chunk text, 2500/300 chunking), same 63-question benchmark (**57 answerable**), same `top_k=20`, cosine. **Only the embedding model changes.** No reranking (dense-only isolates the embedding).

## Source-level (was the right document retrieved?)

| Model | dim | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|--:|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 (baseline) | 384 | 0.877 | 0.965 | 0.983 | 0.983 | 0.912 |
| BAAI/bge-base-en-v1.5 | 768 | 0.877 | 0.983 | 0.983 | 0.983 | 0.915 |
| intfloat/e5-base-v2 | 768 | 0.877 | 1.000 | 1.000 | 1.000 | 0.927 |

## Page-level (was the exact right page retrieved? — citation precision)

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 (baseline) | 0.632 | 0.807 | 0.877 | 0.965 | 0.708 |
| BAAI/bge-base-en-v1.5 | 0.614 | 0.825 | 0.895 | 0.965 | 0.713 |
| intfloat/e5-base-v2 | 0.649 | 0.842 | 0.930 | 0.965 | 0.743 |

## Cost / operational profile

| Model | dim | Index build (s) | Index size (MB) | Query latency avg (ms) | p95 (ms) |
|---|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 (baseline) | 384 | ~6 (Phase 4) | 4.6 | 20.0 | 22.6 |
| BAAI/bge-base-en-v1.5 | 768 | 55.1 | 5.0 | 65.8 | 79.7 |
| intfloat/e5-base-v2 | 768 | 56.9 | 5.1 | 61.5 | 79.5 |

> Latencies are CPU, in-process, on this machine; query latency ≈ query-embedding time (ANN search over 184 vectors is negligible). Index build time and query latency scale with model size / dimension.
