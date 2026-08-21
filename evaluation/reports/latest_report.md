# Evaluation Report — latest

> Auto/hand-maintained status of the evaluation program. Metric tables
> (Recall@K, MRR, faithfulness, …) appear here from Phase 5 onward. This file
> currently records Phase 4 completion and the environment blockers found.

**Updated:** 2026-08-21 · **Branch:** `laukik-uniassist-branch`

---

## Phase 4 — Build the Evaluation Pipeline ✅

The evaluation framework is implemented and runs end-to-end against the frozen
corpus and the 63-question benchmark. See [`../README.md`](../README.md) for
architecture and usage.

**What was verified (real, reproducible):**

- Corpus freeze: **14 files → 144 pages → 184 chunks** (`snapshots/baseline_v1`).
- Isolated local index: **184 vectors, 384-dim (all-MiniLM-L6-v2)**, built in ~6 s.
- Retrieval over 63 questions: **avg 13.6 ms** (p95 16.1 ms) per query.
- Sanity (not a metric): expected source appears in **retrieved top-k for 56/57**
  answerable questions — dense retrieval is healthy.
- Runners emit well-formed raw artifacts with full provenance; the pipeline
  degrades gracefully when external APIs fail.

**Artifacts produced (smoke/demonstration, not the official baseline):**
`experiments/results/baseline_v1_retrieval_raw.json`,
`experiments/results/baseline_v1_full_rag_raw.json`.

> These are **not** the Phase 6 baseline: the index holds raw text (LLM
> summaries unavailable) and reranking was rate-limited. The official
> `results/baseline_v1.json` is produced in Phase 6 once the blockers below are
> resolved and Phase 5 metrics exist.

---

## ⚠ Blockers before a faithful Baseline V1 (Phase 6)

| Blocker | Effect | Needed to resolve |
|---|---|---|
| OpenRouter LLM `402 Payment Required` | No chunk summaries (indexing), no LLM answers | Fund/replace the LLM key, **or** decide to index raw text |
| Cohere Trial key `429` (10/min) | Reranking rate-limited → falls back to retrieval order | Upgrade Cohere key, **or** throttle the eval reranker, **or** run reranker eval later (Phase 9) |

Retrieval-only metrics (Phase 5) can be produced now on the MiniLM index without
either key; the reranked/answer-quality numbers need the keys.

---

## Next

- **Phase 5** — implement `metrics/retrieval_metrics.py` (Recall@1/5/10/20, MRR,
  Precision@K, Hit Rate@K; source- and page-level) and score the raw artifacts.
- **Phase 6** — produce the official `baseline_v1.json` once keys/decisions land.
