# Evaluation Report — latest

> Auto/hand-maintained status of the evaluation program. Metric tables
> (Recall@K, MRR, faithfulness, …) appear here from Phase 5 onward. This file
> records Phases 4-10 completion, the retrieval + reranked baseline numbers, the
> embedding + reranker + chunking selections, and the environment blockers found.

**Updated:** 2026-08-25 · **Branch:** `laukik-uniassist-branch`

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

## Phase 6 — Baseline V1 reference numbers ✅ (retrieval+rerank) · ⚠ (generation partial)

Produced `experiments/results/baseline_v1.json`. **Retrieval and reranking are
complete and faithful** to the frozen Baseline V1 (same corpus/chunking, embedding
`all-MiniLM-L6-v2`, reranker Cohere `rerank-v3.5`); the Cohere key now works, so
the reranker's **true contribution** is measured for the first time (Phase 5 had it
disabled). Full tables: [`retrieval_metrics_baseline_v1_gemini36.md`](./retrieval_metrics_baseline_v1_gemini36.md).

**Reranker contribution — dense vs Cohere rerank-v3.5 (57 answerable):**

| Metric | Dense (Phase 5) | + Cohere rerank | Δ |
|---|---:|---:|---:|
| source Recall@1 | 0.877 | **0.930** | +0.053 |
| source Recall@5 | 0.965 | **0.982** | +0.018 |
| source MRR | 0.912 | **0.953** | +0.041 |
| source P@5 | 0.632 | **0.716** | +0.084 |
| **page Recall@1** | 0.632 | **0.895** | **+0.263** |
| page Recall@5 | 0.807 | **0.965** | +0.158 |
| page MRR | 0.708 | **0.924** | +0.216 |

- **Reranking clearly helps**, and the **page-level** gain is dramatic (Recall@1
  0.63 → 0.90): Cohere surfaces the *exact right page* at rank 1 far more often.
  Direct evidence the reranker is worth its ~640 ms (relevant to citations, Phase 20).
- Reranking was applied on **63/63** questions (no rate-limit fallback) via an
  eval-only trial throttle + 429-retry added to the harness.

**Latency (ms):** retrieval avg **31.6** (p95 40) · reranking avg **640** (p95 1005)
· generation avg **~5000** (p95 7208, **n=16** real answers; gemini-3.6-flash
"thinking" + first-call warmup).

**⚠ Generation is only partially captured (16/63).** Two deviations from the frozen
spec, both documented in `baseline_v1.json`:

1. **LLM substitution** — the frozen `google/gemini-2.5-flash` returns 404 ("no longer
   available to new users") on the available Google key; Google's named successor
   **`gemini-3.6-flash`** was used instead. Retrieval/rerank are unaffected (LLM-independent).
2. **Free-tier daily cap** — `gemini-3.6-flash` free tier allows **20 requests/day**.
   After the cap, `generate_answer` fell back to extractive text (`has_llm=false`).
   A full 63-question answer set needs a paid tier (or ~4 days at 20/day).

**Committed artifacts:** `experiments/results/baseline_v1.json` (reference),
`experiments/results/baseline_v1_gemini36_retrieval_scored.json` (metrics),
`reports/retrieval_metrics_baseline_v1_gemini36.md` (tables). The full-RAG raw dump
(all 63 answers: 16 real + 47 extractive fallback) is left uncommitted per convention
— but note the 16 real answers are **not cheaply reproducible** while the free-tier cap
stands, so keep the local copy until answer-quality is re-run on a paid tier.

---

## Phase 7 — Embedding model experiments ✅

Compared `all-MiniLM-L6-v2` (baseline) vs `BAAI/bge-base-en-v1.5` vs
`intfloat/e5-base-v2` on the **frozen `baseline_v1` snapshot** (raw text, 2500/300),
same 63-question benchmark, same `top_k=20`, dense-only (no rerank) to isolate the
embedding. E5/BGE query/passage instruction prefixes are wired into the eval harness
(`_PrefixedEmbeddings`). Full tables: [`EMBEDDING_COMPARISON.md`](./EMBEDDING_COMPARISON.md);
decision: [`EMBEDDING_DECISION.md`](./EMBEDDING_DECISION.md).

| Model | dim | src Recall@5 | src MRR | page Recall@1 | page MRR | query ms |
|---|--:|--:|--:|--:|--:|--:|
| all-MiniLM-L6-v2 | 384 | 0.965 | 0.912 | 0.632 | 0.708 | 20 |
| bge-base-en-v1.5 | 768 | 0.983 | 0.915 | 0.614 | 0.713 | 66 |
| **e5-base-v2** | 768 | **1.000** | **0.927** | **0.649** | **0.743** | 62 |

- **E5-base-v2 wins on every quality metric** — perfect source Recall@5, best MRR
  (source + page), best page-level recall (57/57 expected sources in top-k). **Recommended.**
- BGE-base is only marginally above MiniLM and not worth its cost here.
- MiniLM is already near-ceiling and ~3× faster; the corpus is small/clean so headroom is limited.
- **Promotion** to production is deferred to Phase 22 (per plan) and needs E5's
  `query:`/`passage:` prefixes wired into `embedder.py`/`retriever.py` + a full re-index.

---

## Phase 8 — Embedding model selection ✅

Finalized with the reranked confirmation (constant Cohere `rerank-v3.5`, 63/63):

| Reranked (top-5) | MiniLM | E5 |
|---|--:|--:|
| source Recall@5 | 0.982 | **1.000** |
| source MRR | 0.953 | **0.962** |
| source Recall@1 | 0.930 | 0.930 |
| page R@1/R@5/MRR | 0.895/0.965/0.924 | 0.895/0.965/0.924 |

**Selected: `intfloat/e5-base-v2`** — E5 ≥ MiniLM at every level (source better, page tied),
so the advantage is robust to reranking. Decision: [`EMBEDDING_DECISION.md`](./EMBEDDING_DECISION.md).
Production promotion is Phase 22 (after the remaining experiments); this phase records the
selection only and does not touch production.

---

## Phase 9 — Reranker A/B ✅ (keep Cohere)

Formalized from existing data (dense top-5 vs Cohere-reranked top-5, both embeddings). Reranking
gives a large, consistent lift — especially **page-level** (Recall@1 ~0.63–0.65 → **0.895**, page MRR
→ **0.924** for both MiniLM and E5) and source Recall@1 0.877 → 0.930 — for ~640 ms/query (trivial vs
LLM latency). **Decision: keep the reranker.** See [`RERANKER_DECISION.md`](./RERANKER_DECISION.md).

## Phase 10 — Chunking experiments ✅ (select 1500/200)

Swept 1000/150 · 1500/200 · 2500/300 (current) · 3500/400 on the selected embedding `e5-base-v2`
(dense, one fresh snapshot per size). Framework change: `chunker.py`/`snapshot.py` now accept
`chunk_size`/`chunk_overlap` (backward-compatible; production default 2500/300 unchanged).

| chunk/overlap | #chunks | src R@1 | src MRR | page R@1 | page MRR |
|---|--:|--:|--:|--:|--:|
| 1000/150 | 384 | 0.912 | 0.939 | 0.702 | 0.776 |
| **1500/200** | 285 | **0.930** | **0.952** | **0.789** | **0.819** |
| 2500/300 (current) | 184 | 0.877 | 0.927 | 0.649 | 0.743 |
| 3500/400 | 159 | 0.860 | 0.917 | 0.649 | 0.741 |

- **1500/200 selected** — best source R@1/MRR and a big **page-level** jump (R@1 0.649 → 0.789).
  Larger chunks lose to E5's 512-token (~2000-char) truncation; no category regresses (table/policy/
  multi-condition stay perfect; exact_terminology/placement/conversational improve). See
  [`CHUNKING_COMPARISON.md`](./CHUNKING_COMPARISON.md) / [`CHUNKING_DECISION.md`](./CHUNKING_DECISION.md).

---

## Environment blockers — status

| Blocker | Status | Notes |
|---|---|---|
| Cohere Trial `429` (10/min) | ✅ **worked around** | eval harness throttle (`EVAL_COHERE_MIN_INTERVAL_MS`) + 429-retry → 63/63 reranked, no fallback |
| OpenRouter LLM `402` | ⛔ still unfunded | account never purchased credits; abandoned in favour of the Google key |
| `gemini-2.5-flash` 404 | ⛔ unavailable | deprecated for new accounts → substituted `gemini-3.6-flash` |
| `gemini-3.6-flash` free tier | ✅ **mitigated** | 20 req/day cap; a **Groq `openai/gpt-oss-120b` fallback** now auto-fails-over in `generator.py`, so answers no longer degrade to extractive after the cap. A full 63-answer re-capture can be run on demand. |

---

## Next

- **Phase 11 (parser)** — evaluate Unstructured.io vs the current pdfplumber/pypdf parser
  (parsing quality + RAG impact). Needs a new dependency; retrieval half needs no LLM.
- **Phase 12–14 (LLM + generation + human eval)** — full 63-answer capture now works via the
  Gemini→Groq failover; can be run on demand (~15 min throttled), then score faithfulness/correctness.
- **Phase 22 (promotion)** — pending config so far: embedding `e5-base-v2`, chunk **1500/200**,
  keep Cohere rerank. Promote (wire E5 prefixes + set chunk size + re-index) only once the suite
  (parser, LLM) is done.
