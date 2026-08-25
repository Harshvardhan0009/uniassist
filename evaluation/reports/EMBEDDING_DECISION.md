# Embedding model decision (Phases 7–8)

**Date:** 2026-08-25 · **Status:** recommendation (production promotion pending approval) · **Dataset:** `uniassist_eval_v1` (57 answerable) · **Method:** dense retrieval on the frozen `baseline_v1` snapshot, `top_k=20`, cosine, no rerank — only the embedding model changes.

Full tables: [`EMBEDDING_COMPARISON.md`](./EMBEDDING_COMPARISON.md). Raw: `experiments/results/embedding_comparison.json`.

## Candidates

| Model | dim | src Recall@1 | src Recall@5 | src MRR | page Recall@1 | page MRR | build (s) | index (MB) | query (ms) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 (baseline) | 384 | 0.877 | 0.965 | 0.912 | 0.632 | 0.708 | ~6 | 4.6 | 20 |
| BAAI/bge-base-en-v1.5 | 768 | 0.877 | 0.983 | 0.915 | 0.614 | 0.713 | 55 | 5.0 | 66 |
| **intfloat/e5-base-v2** | 768 | 0.877 | **1.000** | **0.927** | **0.649** | **0.743** | 57 | 5.1 | 62 |

## Findings

- **E5-base-v2 is the best retriever on this corpus** — perfect source Recall@5 (1.000; the right document is always in the top-5), the best MRR at both source (0.927) and page (0.743) level, and the best page-level recall (relevant to citation precision, Phase 20). It also found the expected source for **57/57** answerable questions in top-k (MiniLM/BGE: 56/57).
- **BGE-base is only marginally better than MiniLM** here (source Recall@5 0.983 vs 0.965) and actually slightly worse at page Recall@1 — not worth 768-dim + ~3× latency on its own.
- **MiniLM is already near-ceiling and much cheaper** (384-dim, ~3× faster query, ~9× faster indexing). On a small, clean corpus its headroom is small.
- **Index size is not a differentiator at this scale** (184 vectors): 4.6 → 5.1 MB. It would matter at millions of vectors (768-dim ≈ 2× vector storage).

## Recommendation

**Adopt `intfloat/e5-base-v2`** as the embedding model, subject to the two checks below. Rationale for the UniAssist workload:

- Retrieval quality is the embedding's core job, and E5 wins on every quality metric, most importantly **page-level** (better citations) and **perfect source Recall@5**.
- The latency cost (~40 ms extra per query) is **negligible** next to LLM generation (~5 s) and reranking (~640 ms). Index build time (57 s) is a one-off offline cost.
- E5 **requires** `query:` / `passage:` prefixes; these are wired into the eval harness and must also be wired into production `retriever.py`/`embedder.py` if promoted (see below).

### Before promoting to production (gated — not done yet)
1. **Confirm the gain survives reranking + end-to-end.** These are dense-only numbers; re-run the reranked comparison (Cohere on top) for MiniLM vs E5 to confirm E5's better candidates translate to better final top-5. (Dense-only already favors E5, and better candidates can only help the reranker.)
2. **Wire E5 prefixes into production.** Production `embedder.py`/`retriever.py` currently embed without instruction prefixes; E5 needs `passage:` at index time and `query:` at query time, or quality drops. This is a code change to promote alongside `EMBEDDING_MODEL=intfloat/e5-base-v2`, plus a **full re-index** of the production Chroma collection.

Per the plan's rules, production is **not** modified until this decision is approved.

## Honest caveats

- Small benchmark (57 answerable, 184 chunks); differences are real but modest at source level. Page-level and Recall@5 gains are the most decision-relevant.
- Snapshot indexes **raw chunk text** (LLM summaries were unavailable in Phase 5/6). If a summary-based index is later adopted, re-run this comparison on that snapshot (the embedding ranking may shift).
