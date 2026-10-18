# Reranker decision (Phase 9)

**Date:** 2026-08-30 · **Status:** ✅ **KEEP Cohere `rerank-v3.5`** (top-20 → top-5). · **Dataset:** `uniassist_eval_v1` (57 answerable). · **Method:** compares **Experiment A** (embedding → top-5, no rerank) vs **Experiment B** (embedding → top-20 → Cohere → top-5), holding the embedding constant. Uses the metrics already produced in Phases 5/6 (MiniLM) and 7/8 (E5) — no new runs.

## Retrieval quality: A (dense top-5) vs B (+ rerank)

| Embedding | metric | A: dense top-5 | B: + Cohere rerank | Δ |
|---|---|--:|--:|--:|
| all-MiniLM-L6-v2 | source Recall@1 | 0.877 | **0.930** | +0.053 |
| | source Recall@5 | 0.965 | **0.982** | +0.017 |
| | source MRR | 0.912 | **0.953** | +0.041 |
| | **page Recall@1** | 0.632 | **0.895** | **+0.263** |
| | page MRR | 0.708 | **0.924** | +0.216 |
| intfloat/e5-base-v2 | source Recall@1 | 0.877 | **0.930** | +0.053 |
| | source Recall@5 | 1.000 | 1.000 | 0.000 |
| | source MRR | 0.927 | **0.962** | +0.035 |
| | **page Recall@1** | 0.649 | **0.895** | **+0.246** |
| | page MRR | 0.743 | **0.924** | +0.181 |

## Cost

- **Rerank latency:** ~**640 ms** avg (p95 ~1005 ms) per query — Cohere API round-trip on top-20.
- Negligible next to LLM generation (~5 s). No local compute/memory cost (hosted API).
- Operational: one external dependency (Cohere) + key; the pipeline already **falls back to retrieval
  order** if the reranker errors, so it fails safe.

## Decision

**Keep the reranker.** It delivers a large, consistent improvement — most importantly at **page level**
(Recall@1 ~0.63–0.65 → **0.895**, page MRR ~0.71–0.74 → **0.924** for both embeddings), which directly
improves citation precision (Phase 20). Source-level Recall@1 also rises 0.877 → 0.930 regardless of
embedding, and MRR improves. The ~640 ms cost is well worth it given LLM latency dominates.

> Note: both embeddings converge to identical **reranked** page metrics (0.895 / 0.924) — once the
> right page is anywhere in the top-20, the cross-encoder reliably lifts it. So the reranker also
> reduces the system's sensitivity to the embedding choice at the final top-5.
