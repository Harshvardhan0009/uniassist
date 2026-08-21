# Evaluation Report — latest

> Auto/hand-maintained status of the evaluation program. Metric tables
> (Recall@K, MRR, faithfulness, …) appear here from Phase 5 onward. This file
> records Phases 4-5 completion, the first retrieval numbers, and the
> environment blockers found.

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

## Phase 5 — Retrieval Metrics ✅

Implemented `metrics/retrieval_metrics.py` (Recall@1/5/10/20, MRR, Precision@K,
Hit Rate@K, source-level set-recall; **source** and **page** relevance levels;
every formula documented in the module). Scored the **dense MiniLM retrieval**
(`--no-rerank`, key-free, fully reproducible) over the **57 answerable** questions
(6 unanswerable excluded). Full tables: [`retrieval_metrics_baseline_v1.md`](./retrieval_metrics_baseline_v1.md).

**Headline — dense retrieval (all-MiniLM-L6-v2, top-20):**

| Level | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---:|---:|---:|---:|---:|
| source | 0.877 | 0.965 | 0.983 | 0.983 | 0.912 |
| page | 0.632 | 0.807 | 0.877 | 0.965 | 0.708 |

**Findings that shape later phases:**

- **Source retrieval is strong** — the right document is in the top-5 for 96.5% of
  questions (MRR 0.91). This is the reference bar BGE/E5 must beat in Phase 7.
- **Conversational is the weakest category** (Recall@1 0.50, Recall@5 0.67,
  MRR 0.58) — expected, since follow-ups are retrieved without history. Direct
  evidence for **Phase 15 (query rewriting)**.
- **exact_terminology** Recall@1 0.71 (abbreviations/codes) — motivates
  **Phase 17 (hybrid dense+sparse)**.
- **Page-level < source-level** (Recall@1 0.63 vs 0.88): the right file ranks
  first, but the exact page less often — relevant to citation precision (Phase 20)
  and chunking (Phase 10).
- Caveat: index currently holds **raw chunk text** (summaries unavailable), and
  MiniLM truncates ~256 tokens, so long chunks lose their tail. Re-scoring on a
  summary-based index is part of Phase 6/7 once the LLM key is funded.

**Committed:** `experiments/results/baseline_v1_retrieval_scored.json` (metrics),
`reports/retrieval_metrics_baseline_v1.md` (tables). The 569 KB raw ranking dump
is regenerable (`retrieval_eval --no-rerank`) and left uncommitted.

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

- **Phase 6** — produce the official `baseline_v1.json` (retrieval metrics +
  reranked metrics + full-RAG latency + captured answers) once the LLM key is
  funded (summaries + answers) and Cohere reranking is available.
- The Phase 5 dense numbers above are the reference bar for **Phase 7** (BGE/E5
  embedding experiments), which reuse this exact frozen snapshot and dataset.
