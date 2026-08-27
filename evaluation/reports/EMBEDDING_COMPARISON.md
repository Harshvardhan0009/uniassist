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

## With Cohere rerank-v3.5 (top-5) — does the dense advantage survive?

Same held-constant reranker applied on top (63/63, no fallback). BGE was eliminated at the
dense stage, so only the winner (E5) vs the baseline (MiniLM) is confirmed here.

| Model | src Recall@1 | src Recall@5 | src MRR | src P@5 | page Recall@1 | page Recall@5 | page MRR |
|---|--:|--:|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 | 0.930 | 0.982 | 0.953 | 0.716 | 0.895 | 0.965 | 0.924 |
| **intfloat/e5-base-v2** | 0.930 | **1.000** | **0.962** | **0.730** | 0.895 | 0.965 | 0.924 |

> With the reranker in the loop, **E5 keeps its source-level edge** (perfect Recall@5, higher MRR).
> Page-level converges (identical) — the cross-encoder reorders the top-20 similarly once both
> retrievers surface the right page within the candidate set. Net: E5 ≥ MiniLM at every level.

## Cost / operational profile

| Model | dim | Index build (s) | Index size (MB) | Query latency avg (ms) | p95 (ms) |
|---|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 (baseline) | 384 | ~6 (Phase 4) | 4.6 | 20.0 | 22.6 |
| BAAI/bge-base-en-v1.5 | 768 | 55.1 | 5.0 | 65.8 | 79.7 |
| intfloat/e5-base-v2 | 768 | 56.9 | 5.1 | 61.5 | 79.5 |

> Latencies are CPU, in-process, on this machine; query latency ≈ query-embedding time (ANN search over 184 vectors is negligible). Index build time and query latency scale with model size / dimension.
