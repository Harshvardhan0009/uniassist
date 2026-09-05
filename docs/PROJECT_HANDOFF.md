# UniAssist / LPUAssist — Project Handoff & Context

> **Read this first.** This is the single source of truth for picking up work on this project in a
> new chat. It captures what the project is, the evaluation program we're running, everything done
> so far, the current configuration, important gotchas, and what's next. When starting a new session,
> just say *"read `docs/PROJECT_HANDOFF.md`"* and you'll have full context.
>
> **Last updated:** 2026-09-05 · **Branch:** `laukik-uniassist-branch` · **HEAD:** `4697de7`
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

## 3. Current production configuration (frozen "Baseline V1")

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

**Why 1500/200 wins:** E5-base truncates at 512 tokens (~2000 chars), so 2500–3500-char chunks lose
their tail. 1500/200 balances completeness vs page-granularity; no category regresses.

### ⏭ AHEAD (not started)

| Phase | What | Notes / blockers |
|---|---|---|
| 11 | **Parser** — Unstructured.io vs current pdfplumber/pypdf | needs new dependency; retrieval half needs no LLM |
| 12 | **LLM comparison** (same retrieved context, swap LLM) | Gemini vs Groq gpt-oss vs others; failover makes capture feasible |
| 13 | **Generation metrics** — faithfulness, answer correctness, citation correctness, hallucination rate (esp. unanswerable) | run full_rag then score; ~15 min throttled |
| 14 | **Human eval** (~30–50 Qs, 1–5 scores) | validate auto-metrics |
| 15 | Query rewriting (conversational) | conversational is the weakest category |
| 16 | Metadata filtering | add domain metadata → filtered retrieval |
| 17 | Hybrid search (dense + sparse/BM25) | needs `rank-bm25` |
| 18 | Abstention / "I don't know" robustness | currently prompt-only |
| 19 | Document versioning | prefer current policies |
| 20 | Improve citations (page + excerpt in UI) | page-level metrics already tracked |
| 21 | `FINAL_MODEL_SELECTION.md` | consolidate all decisions |
| 22 | **Promote winning config to production** | see §7 |
| 23 | Observability / tracing | optional LangSmith |
| 24 | Production hardening | S3/Supabase/Redis/Celery/auth (roadmap) |
| 25 | Final documentation | update ARCHITECTURE/README |

---

## 7. Emerging winning configuration (promotion deferred to Phase 22)

Backed by evaluation so far — **NOT yet in production**:

- **Embedding:** `intfloat/e5-base-v2` (768-dim) — *needs `query:`/`passage:` prefixes wired into
  `embedder.py`/`retriever.py` + a full 768-dim re-index of production Chroma.*
- **Chunking:** **1500 / 200** (change `chunker.py` constants or drive from config + re-index).
- **Reranker:** keep **Cohere rerank-v3.5**, top-20 → top-5.
- **LLM:** primary `gemini-3.6-flash` + Groq fallback (revisit in Phase 12).

**Promotion is one coordinated migration** (prefixes + chunk size + re-index) and only happens after
Phases 11–21 are done and approved.

---

## 8. How to resume in a new chat

1. Say: *"read `docs/PROJECT_HANDOFF.md`"* (this file) and *"read `evaluation/reports/latest_report.md`"*
   for the freshest numbers.
2. Confirm git state: `git log --oneline -5`, `git status`.
3. Pick the next phase (default: **Phase 11 parser**, or **Phases 12–14 answer quality**).
4. Follow the golden rules (§2): one variable, same dataset/corpus, record latency, don't touch
   production, commit + push curated artifacts (scan for secrets first), update the docs
   (`latest_report.md`, `README.md`, `ARCHITECTURE.md`, and this handoff).
5. **Keep this file current** at the end of each phase (update "Last updated", HEAD, the progress
   tables, and the winning config).

---

## 9. Commit history (evaluation program)

```
4697de7 Phases 9-10 - keep reranker; select 1500/200 chunking
f1c4c6e Phase 8 embedding selection - e5-base-v2 (reranked-confirmed)
e627dcc Groq fallback LLM (primary -> fallback -> extractive)
af2ce01 Phase 7 embedding comparison - e5-base-v2 wins
a387d8d Phase 6 baseline - Cohere-reranked metrics + Gemini answers (partial)
e224c52 Phase 5 retrieval metrics + dense baseline
c98434f Phase 4 evaluation pipeline framework
c23678e Phase 3 ground truth
d68a9c1 Phase 2 benchmark dataset
4583efe Phase 1 freeze Baseline V1
77fc1f6 Phase 0 codebase audit
```
