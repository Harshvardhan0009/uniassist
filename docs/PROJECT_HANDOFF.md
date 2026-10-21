# UniAssist / LPUAssist — Project Handoff & Context

> **Read this first.** This is the single source of truth for picking up work on this project in a
> new chat. It captures what the project is, the evaluation program we're running, everything done
> so far, the current configuration, important gotchas, and what's next. When starting a new session,
> just say *"read `docs/PROJECT_HANDOFF.md`"* and you'll have full context.
>
> **Last updated:** 2026-10-21 · **Branch:** `laukik-uniassist-branch` · **HEAD:** `57db9e3`
> · **Remote:** `github.com/Harshvardhan0009/uniassist`

---

## 1. What this project is

**UniAssist** (branded **LPUAssist** in the UI) is a **RAG assistant** that answers questions about
Lovely Professional University policies (dress code, placements, library, academic benefits,
residential facilities, semester exchange, NSS, campus map), grounded strictly in official university
documents.

**Two deployable parts:**
- **Backend** — FastAPI (`backend/app/`). Offline **ingestion** (documents → chunks → summaries →
  embeddings → ChromaDB) + online **query** (retrieve → rerank → generate).
- **Frontend** — Next.js 16 / React 19 single-page chat UI (`frontend/app/page.js`).

**Live pipeline:** `question → embed → ChromaDB similarity (top-20) → Cohere rerank (top-5) →
LLM answer + citations`.

**Key docs:**
- `ARCHITECTURE.md` — full current architecture (kept up to date).
- `docs/CURRENT_STATE.md` — Phase 0 audit + a §13 "status updates" log.
- `evaluation/README.md` — the evaluation framework + phase status table.
- `evaluation/reports/latest_report.md` — **living status report** with all metrics (read this for
  the latest numbers).
- Decision docs: `evaluation/reports/{EMBEDDING,RERANKER,CHUNKING}_DECISION.md`.

---

## 2. The mission (why we're doing this)

Turn UniAssist from a working prototype into an **evaluation-driven, production-quality RAG system**.
Every component choice (embedding, reranker, chunking, parser, LLM) must be backed by **reproducible
measurement**, not assumptions. We work **phase-by-phase** from a 25-phase master plan and **do not
modify production** until the full winning configuration is chosen (that's Phase 22).

**Golden rules we follow:**
1. **One variable at a time** per experiment.
2. **Same benchmark dataset** (`uniassist_eval_v1`, 63 questions) and **same corpus** every time.
3. **Record latency**, not just quality.
4. **Never delete previous results** (reproducibility).
5. **Don't touch production** until the experiment suite is done (Phase 22).
6. Work phase-by-phase; explain what/why, run, show results, **wait for approval** before the next
   major phase.

---

## 3. Frozen Baseline V1 (historical)

> **Historical Reference Note:** This section records the original frozen Baseline V1 reference configuration
> established in Phase 1 for benchmark comparisons. The **current live production configuration** promoted
> in Phase 22 is detailed in [Section 7](#7-production-architecture-promoted-in-phase-22) below.

| Component | Value |
|---|---|
| Parser | pdfplumber (table-aware) + pypdf fallback; python-docx; TextLoader |
| Chunking | RecursiveCharacterTextSplitter, **2500 / 300** chars (keep-whole if ≤ 2500) |
| Embedding | **all-MiniLM-L6-v2** (384-dim, normalized, cosine), in-process |
| Vector store | ChromaDB, collection `university_docs` (remote Render, client-server) |
| Retrieval | top-K = **20** |
| Reranker | Cohere **rerank-v3.5**, top-N = **5** |
| LLM (primary) | **`gemini-3.6-flash`** via Google's OpenAI-compatible endpoint |
| LLM (fallback) | **Groq `openai/gpt-oss-120b`** (auto-failover) |

**Corpus:** 14 ingestible files (12 PDF + 1 DOCX + 1 TXT) → 144 pages → **184 chunks** at 2500/300.

---

## 4. Important environment facts / gotchas (READ before running)

- **Python venv:** `backend\venv\Scripts\python.exe` (Python 3.13.1). Run eval modules from the
  **project root**, e.g. `& "backend\venv\Scripts\python.exe" -m evaluation.runners.retrieval_eval ...`.
- **OS:** Windows, PowerShell. Console is cp1252 — code forces UTF-8 stdout, but ad-hoc `python -c`
  printing unicode (₹, arrows) can still choke; use temp scripts with `sys.stdout.reconfigure`.
- **Secrets** live only in `backend/.env` (gitignored). **Never commit keys.** Always secret-scan
  staged diffs for `gsk_`/`AQ.` before committing.
- **LLM history / why the substitutions:**
  - OpenRouter key is **unfunded** (`402`).
  - Frozen baseline model `gemini-2.5-flash` returns **404 "no longer available to new users"** →
    substituted **`gemini-3.6-flash`**.
  - `gemini-3.6-flash` **free tier caps at 20 requests/day** → this limited Phase 6 answer capture.
    **Mitigated** by the Groq fallback (`generator.py` fails over primary → fallback → extractive).
- **Cohere key is Trial-tier (10 calls/min).** For reranked eval runs, set
  `$env:EVAL_COHERE_MIN_INTERVAL_MS=7000` — the harness throttles + retries on 429. A full
  63-question reranked run takes ~15 min because of this.
- **Reasoning models return empty at tiny `max_tokens`** (Gemini 3.x, gpt-oss "think" first). Use a
  realistic budget (≥256).
- **E5/BGE need instruction prefixes.** E5 **requires** `query:` / `passage:`; BGE uses a query
  instruction. Wired into the eval harness via `_PrefixedEmbeddings` (config `embedding.query_prefix`
  / `passage_prefix`). **Production `embedder.py`/`retriever.py` do NOT apply prefixes yet** — that's
  part of the Phase 22 E5 promotion.
- **Config gotcha (fixed):** experiment configs must put `snapshot`/`collection` under the
  **`"evaluation"`** block, not `"experiment"` (`load_config` reads `evaluation`). A chunk run once
  silently reused the wrong snapshot because of this.

---

## 5. The evaluation framework (how experiments work)

Under `evaluation/`. Design guarantees: **no production mutation** (isolated local Chroma under
`evaluation/.chroma/`), **frozen snapshot** re-embedded identically per experiment, **raw records**
separate from **metric scoring**.

```
Data/ → snapshot.py (freeze partition→chunk→[summarize]) → snapshots/<name>.jsonl (+manifest)
      → indexer.py (embed → isolated local Chroma) → .chroma/<collection>
questions.json (63 Qs + ground truth) + store → runners/{retrieval_eval,full_rag_eval}.py
      → experiments/results/*.json (raw) → metrics/retrieval_metrics.py → scored + reports/*.md
```

**Common commands (from project root):**
```powershell
# Dense retrieval experiment (no rerank), then score it:
& "backend\venv\Scripts\python.exe" -m evaluation.runners.retrieval_eval --config <cfg> --no-rerank
& "backend\venv\Scripts\python.exe" -m evaluation.metrics.retrieval_metrics --result evaluation\experiments\results\<cfg>_retrieval_raw.json

# Reranked / full-RAG run (throttle Cohere Trial key):
$env:EVAL_COHERE_MIN_INTERVAL_MS=7000
& "backend\venv\Scripts\python.exe" -m evaluation.runners.full_rag_eval --config baseline_v1_gemini36
```

**Metrics:** Recall@1/5/10/20, MRR, Precision@K, at **source level** (right document) and **page
level** (right page — matters for citations). 57 of 63 questions are answerable (6 unanswerable).

**Configs live in** `evaluation/configs/*.json`. **Committed convention:** curated artifacts
(configs, scored metrics, comparison JSON, reports) are committed; large **raw** dumps
(`*_retrieval_raw.json`, `*_full_rag_raw.json`) are regenerable and **left uncommitted**.

---

## 6. Phase-by-phase progress

### ✅ DONE

| Phase | What | Result / artifact |
|---|---|---|
| 0 | Codebase audit | `docs/CURRENT_STATE.md` |
| 1 | Freeze Baseline V1 | `docs/BASELINE_V1.md`, `evaluation/configs/baseline_v1.json` |
| 2 | Benchmark dataset (63 Qs, 9 categories) | `evaluation/dataset/questions.json` |
| 3 | Ground truth (expected sources/pages/answers) | same dataset, `-phase3` |
| 4 | Evaluation pipeline (snapshot/indexer/harness/runners) | `evaluation/lib/`, `runners/` |
| 5 | Retrieval metrics + dense MiniLM baseline | src Recall@5 **0.965**, MRR **0.912** |
| 6 | Official baseline (+ Cohere rerank + LLM answers) | `results/baseline_v1.json`; reranked src Recall@5 **0.982**, MRR **0.953**, page R@1 0.63→**0.90**. Answers 16/63 (Gemini cap; now mitigated by Groq) |
| — | **Groq fallback LLM** | `generator.py` primary→fallback→extractive; verified |
| 7 | Embedding comparison (MiniLM vs BGE vs E5) | **e5-base-v2 wins**: dense src Recall@5 **1.000**, MRR **0.927** |
| 8 | Embedding **selection** | **`intfloat/e5-base-v2`** (confirmed under reranking: src Recall@5 1.000 / MRR 0.962). `EMBEDDING_DECISION.md` |
| 9 | Reranker A/B | **Keep Cohere** (page R@1 ~0.63→0.90 for ~640ms). `RERANKER_DECISION.md` |
| 10 | Chunking sweep (1000/1500/2500/3500) on E5 | **1500/200 selected**: src R@1 0.877→**0.930**, page R@1 0.649→**0.789**. `CHUNKING_DECISION.md` |
| 11 | Parser A/B (Unstructured.io vs current `pdfplumber`) | **Keep current `pdfplumber` (table-aware)**: higher dense src R@1 (**0.930** vs 0.895), page R@1 (**0.789** vs 0.737), 2x faster ingestion, no heavy dependencies. `PARSER_DECISION.md` |
| 12 | LLM comparison (Gemini 3.6 Flash vs Groq GPT-OSS-120B) | **Retain Gemini 3.6 Flash primary, Groq GPT-OSS-120B fallback**: on matched 18Q answerable subset (Q001–Q018), Gemini +29.8% token recall, +8.7% numerical recall, 3.1× faster. Groq evaluated on full 63Q with 6/6 unanswerable abstention. See [`LLM_DECISION.md`](../evaluation/reports/LLM_DECISION.md) and [`LLM_MATCHED_SUBSET_ANALYSIS.md`](../evaluation/reports/LLM_MATCHED_SUBSET_ANALYSIS.md) |
| 13 | Generation metrics | **Automated scoring suite implemented**: Citation F1, token recall/F1, numerical recall, unanswerable abstention rate (`metrics/generation_metrics.py`). Scored baseline, Gemini, and Groq runs. |
| 14 | Human eval (30-question stratified sample) | **4.68/5.00 overall score; validates auto-metrics**: Faithfulness 4.80, Correctness 4.53, Citation 4.67, Clarity 4.73, Abstention 100%. Highlighted conversational context vulnerability (3.62/5.00) driving Phase 15. `HUMAN_EVALUATION.md` |
| 15 | Query rewriting (conversational) | **Adopted**: conversational src R@1 **0.667 → 1.000** (4/6 → 6/6, +50%), page R@1 **0.500 → 1.000** (3/6 → 6/6, +100%; small sample of 6 questions); benchmark dense src R@1 **0.930 → 0.965** (53/57 → 55/57, +2 Qs), page R@1 **0.789 → 0.842** (45/57 → 48/57, +3 Qs). Multi-turn resolution via rewriter before retrieval. `QUERY_REWRITING_DECISION.md` |
| 16 | Metadata filtering | **Rejected automated hard filtering**: Oracle gives 0% gain (src R@1 tied at 0.965, 55/57); automated router misclassifications cause false-negative exclusions (src R@1 drops to 0.947, 54/57, -1 Q). Retain unconstrained retrieval + Cohere reranker; metadata reserved for UI badges. `METADATA_FILTERING_DECISION.md` |
| 17 | Hybrid search (Dense + BM25 RRF) | **Rejected hybrid as default**: identical 1.000 src R@5 (57/57) and 0.965 src R@1 (55/57); page R@1 slightly lower (0.825 vs 0.842, 47/57 vs 48/57, -1 Q); 2.9x latency increase (51 ms vs 17.6 ms). Retain unfiltered Dense E5 + Cohere reranker; tooling preserved in `evaluation/lib/hybrid.py`. `HYBRID_SEARCH_DECISION.md` |
| 18 | Abstention & refusal calibration | **Adopted prompt-based refusal; rejected pre-LLM score gating**: 100% abstention precision & recall (6/6) across Gemini and Groq with 0 hallucinations. Pre-LLM score thresholding rejected because score overlap causes 33.3% false refusals on valid queries at $\tau = 0.80$. `ABSTENTION_DECISION.md` |
| 19 | Context compression & trimming | **Adopted adaptive score-aware pruning & structural cleanup**: 15.9% prompt token reduction (avg 3.89 chunks vs 5.00); 100% source recall preserved (57/57); page-aware headers. `CONTEXT_COMPRESSION_DECISION.md` |
| 20 | Faithfulness & hallucination audit | **Comprehensive audit completed**: 0 phantom citations (0.0%). Original audit reported numerical hallucinations dropping from 28 to 9 (Groq) and 2 (Gemini), but compared unmatched samples (63 vs 63 vs 18 Qs). On the matched 18Q subset, Baseline=8, Groq=2, Gemini=2 (tied). See [`FAITHFULNESS_HALLUCINATION_AUDIT.md`](../evaluation/reports/FAITHFULNESS_HALLUCINATION_AUDIT.md) and [`LLM_MATCHED_SUBSET_ANALYSIS.md`](../evaluation/reports/LLM_MATCHED_SUBSET_ANALYSIS.md) |
| 21 | Latency & cost optimization | **Latency waterfall & token economics mapped**: end-to-end TTFT ~1.4s (streaming), API cost $2.14 / 1,000 queries with Cohere (<$110/mo for 10,000 students). `LATENCY_COST_OPTIMIZATION.md` |
| 22 | **Promote winning config to production** | **Promoted to `backend/app/`**: `intfloat/e5-base-v2` with `_PrefixedEmbeddings`, `1500/200` chunking, Cohere reranking, conversational rewriter, adaptive compressor, calibrated prompt, and main API telemetry. `PROMOTION_TO_PRODUCTION.md` |
| 23 | Observability / tracing | **Structured telemetry implemented**: `backend/app/telemetry.py` + `GET /api/metrics` endpoint. `OBSERVABILITY_TRACING.md` |
| 24 | Production hardening | **Hardening checklist verified**: input validation, rate limiting, constant-time bearer authentication, zero-secret leakage, multi-tier LLM failover. `PRODUCTION_HARDENING.md` |
| 25 | Final documentation | **Documentation synchronized**: `latest_report.md`, `README.md`, `CURRENT_STATE.md`, `PROJECT_HANDOFF.md`. Evaluation program 100% complete. |

---

## 7. Production Architecture (Promoted in Phase 22)

The production codebase in `backend/app/` is now running the fully converged, evaluation-proven architecture:

- **Parser:** `pdfplumber` (table-aware) pipeline (`partition.py`).
- **Embedding:** `intfloat/e5-base-v2` (768-dim) with automatic asymmetric `query: ` and `passage: ` instruction prefixes (`embedder.py`).
- **Chunking:** **1500 / 200** (`chunker.py`).
- **Retrieval:** Unfiltered dense similarity search (Top-20).
- **Reranker:** Cohere `rerank-v3.5`, Top-20 → Top-5 (`reranker.py`).
- **Rewriter:** Multi-turn conversational query reformulation for turn 2+ queries (`rewriter.py`).
- **Compressor:** Adaptive score-aware tail pruning & markdown/table whitespace compaction (`compressor.py`).
- **LLM:** Primary **`gemini-3.6-flash`** + **Groq `openai/gpt-oss-120b`** fallback + deterministic extractive fallback (`generator.py`).
- **Grounding & Refusal:** Calibrated prompt with negative constraints (100% clean refusal on unanswerables; page-aware citations).
- **Observability:** Structured JSON trace logging and real-time operational endpoint `GET /api/metrics` (`telemetry.py`).

---

## 8. Current Next Steps & Maintenance

The 25-phase evaluation program is complete and the winning architecture is in production. Immediate next steps for project sustainability:

1. **Minimal Pytest Smoke Suite (`backend/tests/`):**
   - Offline unit tests for `/api/health`, `/api/metrics`, request parameter validation (`/api/query`), constant-time token bearer auth (`/api/ingest`), chunker splitting boundaries, and context compressor score pruning.
2. **Continuous Integration (`.github/workflows/ci.yml`):**
   - Run the offline test suite on every PR and push without requiring external API keys.
3. **Production Observability & Monitoring:**
   - Monitor `GET /api/metrics` for p50/p95 latency shifts and LLM failover frequency.
   - Configure secrets securely in the production deployment environment.

---

## 9. Commit history (evaluation program)

```
57db9e3 ci: add GitHub Actions workflow for backend offline smoke tests
42c3558 test(backend): add minimal offline pytest smoke suite
dc6a045 docs: reconcile PROJECT_HANDOFF.md and document evaluation limitations in latest_report.md
b64f3b4 docs: reconcile ARCHITECTURE.md with promoted production system
fcfe1a8 docs: synchronize README, ARCHITECTURE, and backend/.env.example with production RAG pipeline
c67d5b9 fix(frontend): support apostrophes in citation regex and ignore backend logs
df3fcda feat(rag): complete 25-phase optimization, promote winning pipeline, reindex vector store, and add real-time streaming UX
3b579fb docs: add PROJECT_HANDOFF.md (project context + phase progress + next steps)
4697de7 feat(eval): Phases 9-10 - keep reranker; select 1500/200 chunking
f1c4c6e docs(eval): Phase 8 embedding selection - e5-base-v2 (reranked-confirmed)
e627dcc feat(llm): add automatic Groq fallback LLM (primary -> fallback -> extractive)
af2ce01 feat(eval): Phase 7 embedding comparison - e5-base-v2 wins (MiniLM/BGE/E5)
a387d8d feat(eval): Phase 6 baseline - Cohere-reranked metrics + Gemini answers (partial)
e224c52 feat(eval): retrieval metrics + dense baseline scored (Phase 5)
c98434f feat(eval): add evaluation pipeline framework (Phase 4)
c23678e feat(eval): attach ground truth to uniassist_eval_v1 (Phase 3)
d68a9c1 feat(eval): add benchmark dataset uniassist_eval_v1 (Phase 2)
4583efe chore(eval): freeze Baseline V1 (Phase 1)
77fc1f6 docs: add Phase 0 codebase audit (CURRENT_STATE.md)
```

---

## 10. Known limitations & evaluation caveats

1. **Phase 6 Answer Coverage vs Later Phases:**
   - In Phase 6 (`baseline_v1.json`), only **16/63** answers were captured with a real LLM due to the Gemini free-tier daily cap (20 requests/day).
   - In subsequent phases, full 63-question answer coverage was achieved by Groq in Phase 12/13 (`rag_e5_1500_groq_gptoss_full_rag_scored.json`, 63/63) and in the Phase 20 audit (`faithfulness_audit_scored.json`, 63/63). The Phase 12/13 Gemini run evaluated 18 questions (`rag_e5_1500_gemini_full_rag_scored.json`) before hitting the daily quota.
2. **Statistical Ties on 57 Answerable Questions:**
   - The benchmark dataset contains 57 answerable questions. A 1-question difference equals $1/57 \approx 1.75\%$.
   - A delta such as 0.965 (55/57) vs 0.947 (54/57) in Recall@1 represents exactly **one question**. Such small deltas represent statistical ties rather than significant margins. Decisions were made considering latency, operational simplicity, and cross-encoder synergy rather than single-question retrieval deltas alone.
3. **Abstention Sample Size:**
   - Out-of-corpus refusal calibration (Phase 18) was evaluated on **6 unanswerable questions** (`Q056`–`Q061`). While achieving 6/6 (100%) clean abstention on both models confirms prompt refusal capabilities, 6 queries is a small sample. Expanding to a 20+ query adversarial set is recommended for future audits.
4. **Human Evaluation Scorer:**
   - The 30-question stratified human evaluation in Phase 14 (`human_eval_30q_scored.json`) produced an overall score of 4.68/5.00. Scorer identity: `TODO: verify` (to be filled in by the author).
5. **Numerical Hallucination Comparison:**
   - The reported reduction in numerical hallucinations (from 28 down to 2) compares the Baseline V1 system (MiniLM, 2500/300, extractive fallbacks, uncalibrated prompt) against the promoted Gemini configuration (E5, 1500/200, context compression, calibrated slab-aware prompt). This is a **system-level comparison**, not a single-variable ablation.
6. **Operational Limits & Cold Starts:**
   - **Google Gemini free-tier:** Capped at 20 requests/day. The live demo depends heavily on automatic failover to the Groq `openai/gpt-oss-120b` fallback when quota is reached.
   - **Cohere Trial Key:** Limited to 10 calls/minute, requiring an eval-only throttle (`EVAL_COHERE_MIN_INTERVAL_MS=7000`) during batch runs.
   - **Hosted Render ChromaDB:** Free/starter container instances spin down after inactivity; initial requests can experience a 30–50+ second cold start while the remote container wakes up.