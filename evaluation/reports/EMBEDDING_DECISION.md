# Embedding model decision (Phases 7–8)

**Date:** 2026-08-27 · **Status:** ✅ **SELECTED — `intfloat/e5-base-v2`** (Phase 8 complete). Production promotion is deferred to **Phase 22** (after the remaining experiments), per the plan. · **Dataset:** `uniassist_eval_v1` (57 answerable) · **Method:** dense retrieval on the frozen `baseline_v1` snapshot, `top_k=20`, cosine — only the embedding model changes; then confirmed with the constant Cohere reranker on top.

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

**Selected: `intfloat/e5-base-v2`** as the embedding model — confirmed by the reranked comparison
above (E5 ≥ MiniLM at every level). Rationale for the UniAssist workload:

- Retrieval quality is the embedding's core job, and E5 wins on every quality metric, most importantly **page-level** (better citations) and **perfect source Recall@5**.
- The latency cost (~40 ms extra per query) is **negligible** next to LLM generation (~5 s) and reranking (~640 ms). Index build time (57 s) is a one-off offline cost.
- E5 **requires** `query:` / `passage:` prefixes; these are wired into the eval harness and must also be wired into production `retriever.py`/`embedder.py` if promoted (see below).

### Reranked confirmation (done)

With the constant Cohere `rerank-v3.5` applied on top (63/63, no fallback), E5 keeps its edge:

| Reranked (top-5) | MiniLM | E5 |
|---|--:|--:|
| source Recall@5 | 0.982 | **1.000** |
| source MRR | 0.953 | **0.962** |
| source Recall@1 | 0.930 | 0.930 |
| page R@1 / R@5 / MRR | 0.895 / 0.965 / 0.924 | 0.895 / 0.965 / 0.924 |

E5 ≥ MiniLM at every level (source better, page tied). **The selection is robust to reranking.**

### Production promotion (deferred to Phase 22 — NOT done)

Promoting E5 to the live system will require, as one coordinated migration:
1. **Wire E5 prefixes into production** `embedder.py`/`retriever.py`: `passage:` at index time and
   `query:` at query time (omitting them badly degrades E5).
2. Set `EMBEDDING_MODEL=intfloat/e5-base-v2` and **fully re-index** the production Chroma collection
   (384-dim → 768-dim; existing MiniLM vectors are dimension-incompatible).

Per the plan (Phase 22), production is **not** modified until all experiments (chunking, parser, LLM,
hybrid, …) are complete and the full winning configuration is chosen. This document records the
**embedding selection** only; it does not change production.

## Honest caveats

- Small benchmark (57 answerable, 184 chunks); differences are real but modest at source level. Page-level and Recall@5 gains are the most decision-relevant.
- Snapshot indexes **raw chunk text** (LLM summaries were unavailable in Phase 5/6). If a summary-based index is later adopted, re-run this comparison on that snapshot (the embedding ranking may shift).
