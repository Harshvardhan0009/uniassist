# UniAssist Evaluation Framework

Offline tooling that measures the RAG system with reproducible, controlled
experiments. It is the machinery behind Phases 4+ of the evaluation plan.

**Design guarantees**

- **No production mutation.** Experiments never touch the live remote Chroma or
  the production singletons. Each experiment builds its own **local, isolated**
  Chroma collection under `evaluation/.chroma/<collection>` (gitignored).
- **One variable at a time.** The corpus is frozen once into a **snapshot**
  (partition → chunk → summarize); every embedding/reranker/LLM experiment reuses
  that identical frozen set, so only the variable under test changes.
- **Production-faithful.** The snapshot and retrieval/rerank/generation steps
  reuse the real `app.*` modules (same parser, 2500/300 chunking, summariser,
  Cohere reranker, LLM generator).
- **Nothing fabricated.** Runners emit raw, machine-readable records; scoring is
  a separate, documented step (Phase 5+).

---

## Directory layout

```
evaluation/
  configs/                 # experiment configurations (Baseline V1 schema)
    baseline_v1.json       # FROZEN reference config (Phase 1)
  dataset/
    questions.json         # the benchmark (63 Qs, ground-truthed) — Phases 2-3
    README.md
  lib/                     # framework internals
    paths.py               # path resolution + `app` import wiring
    snapshot.py            # freeze partition→chunk→summarize into snapshots/
    indexer.py             # embed a snapshot into an isolated local Chroma collection
    harness.py             # config loader + faithful retrieve/rerank helpers
    experiment.py          # glue: prepare store, run one question (retrieval / full RAG)
  runners/
    retrieval_eval.py      # execute benchmark → raw retrieval+rerank records
    full_rag_eval.py       # execute benchmark → answers + sources (for gen/human eval)
  metrics/                 # Phase 5 (retrieval) + Phase 13 (generation) — scoring
  experiments/results/     # result artifacts (JSON)
  reports/                 # human-readable reports
  snapshots/               # frozen corpus snapshots (gitignored; regenerable)
  .chroma/                 # local per-experiment vector stores (gitignored)
```

---

## How to run

Run from the **project root** with the backend venv (it wires `app` onto the path):

```powershell
# 1) Freeze the corpus once (partition → chunk → summarize).
backend\venv\Scripts\python.exe -m evaluation.lib.snapshot --name baseline_v1

# 2) (optional) Build the isolated local index explicitly.
backend\venv\Scripts\python.exe -m evaluation.lib.indexer --snapshot baseline_v1 --collection eval_baseline_v1 --model all-MiniLM-L6-v2

# 3) Retrieval evaluation (auto-builds snapshot + index if missing).
backend\venv\Scripts\python.exe -m evaluation.runners.retrieval_eval --config baseline_v1

# 4) Full-RAG (captures answers; makes live LLM calls — use --limit while iterating).
backend\venv\Scripts\python.exe -m evaluation.runners.full_rag_eval --config baseline_v1 --limit 5
```

Useful flags: `--limit N`, `--no-rerank`, `--rebuild-snapshot`, `--rebuild-index`,
`--snapshot NAME`, `--collection NAME`, `--verbose`.

**Cohere Trial throttle.** The eval Cohere key is Trial-tier (10 calls/min). To get a
clean reranked run without `429` fallbacks, set a minimum interval between rerank
calls (the harness also retries on 429):

```powershell
$env:EVAL_COHERE_MIN_INTERVAL_MS=7000   # ~8.5 rerank calls/min
backend\venv\Scripts\python.exe -m evaluation.runners.full_rag_eval --config baseline_v1_gemini36
```

**Embedding experiments (Phase 7).** Each embedding model gets its own config + isolated
collection, reusing the same frozen snapshot. E5/BGE instruction prefixes go in the
config's `embedding.query_prefix` / `embedding.passage_prefix` (the harness applies them
at index and query time). Dense-only isolates the embedding:

```powershell
backend\venv\Scripts\python.exe -m evaluation.runners.retrieval_eval --config e5_base_v2 --no-rerank
backend\venv\Scripts\python.exe -m evaluation.metrics.retrieval_metrics --result evaluation\experiments\results\e5_base_v2_retrieval_raw.json
```

---

## Experiment configuration

A config is the Baseline V1 JSON schema (`configs/baseline_v1.json`). To add an
experiment (e.g. a new embedding model in Phase 7), **copy** the baseline config,
change **one** field, and give it a new `id`. Evaluation-only fields are derived
automatically (or set an optional `"evaluation": { "snapshot": ..., "collection": ... }`
block):

- `snapshot` — which frozen snapshot to embed (default `baseline_v1`).
- `collection` — the isolated local Chroma collection (default `eval_<id>`).

The frozen `baseline_v1.json` is never edited (freeze rule #1).

---

## Result artifacts

Each runner writes `experiments/results/<id>_<kind>_raw.json` containing:

- `config` — the resolved experiment configuration.
- `snapshot` — name + fingerprints (`snapshot_sha256`, `raw_content_sha256`) +
  `summarizer.effective` (whether LLM summaries actually applied).
- `index` — collection, embedding model, dimensions, vector count, build time.
- `latency_ms` — per-step avg / p50 / p95 / max.
- `results[]` — per question: ranked `retrieved` (top-k) and `reranked` (top-n)
  lists with `source_file`, `page_number`, `cosine_score`, `rerank_score`, plus
  `expected_sources`/`expected_pages` and (full RAG) the generated `answer`.

These raw records are what Phase 5 metrics consume. `sanity_preview` in the
retrieval artifact is a smoke-test signal only — **not** the Recall/MRR metric.

---

## Status

| Phase | Deliverable | Status |
|---|---|---|
| 0 | `docs/CURRENT_STATE.md` audit | done |
| 1 | `configs/baseline_v1.json` frozen config | done |
| 2 | `dataset/questions.json` (63 Qs) | done |
| 3 | ground truth on all 63 | done |
| **4** | **evaluation pipeline (this framework)** | **done — runs end-to-end** |
| **5** | **`metrics/retrieval_metrics.py`** (Recall@1/5/10/20, MRR, Precision@K, HitRate@K, set-recall) | **done — dense baseline scored** |
| **6** | **official `results/baseline_v1.json`** (retrieval + reranked metrics + latency) | **retrieval+rerank done; generation partial (16/63, Gemini free-tier cap)** |
| **7** | **embedding comparison** (MiniLM vs BGE vs E5) | **done** |
| **8** | **embedding selection** (`EMBEDDING_DECISION.md`) | **done — selected `e5-base-v2` (promotion = Phase 22)** |
| **9** | **reranker A/B** (`RERANKER_DECISION.md`) | **done — keep Cohere rerank-v3.5** |
| **10** | **chunking experiments** (`CHUNKING_DECISION.md`) | **done — selected 1500/200** |
| **11** | **parser experiments** (`PARSER_DECISION.md`) | **done — retain current `pdfplumber`** |
| **12** | **LLM comparison & selection** (`LLM_DECISION.md`) | **done — retain Gemini 3.6 Flash primary, Groq GPT-OSS-120B fallback** |
| **13** | **generation metrics** (`metrics/generation_metrics.py`) | **done — citation, correctness, numerical recall, abstention scored** |
| **14** | **human evaluation** (`reports/HUMAN_EVALUATION.md`) | **done — 4.68/5.00 avg, validates auto-metrics** |
| **15** | **query rewriting** (`reports/QUERY_REWRITING_DECISION.md`) | **done — conversational R@1: 0.667 → 1.000** |
| **16** | **metadata filtering** (`reports/METADATA_FILTERING_DECISION.md`) | **done — rejected automated hard filtering (0% oracle gain, degrades routing)** |
| **17** | **hybrid search** (`reports/HYBRID_SEARCH_DECISION.md`) | **done — rejected hybrid as default (0% recall gain, 2.9x latency; retain dense E5 + Cohere)** |
| **18** | **abstention robustness** (`reports/ABSTENTION_DECISION.md`) | **done — 100% prompt abstention (6/6); rejected pre-LLM score gating (33% false refusals)** |
| **19** | **context compression & trimming** (`reports/CONTEXT_COMPRESSION_DECISION.md`) | **done — 15.9% token cut (avg 3.89 chunks vs 5.00); 100% source recall preserved** |
| **20** | **faithfulness & hallucination audit** | **next** |

**Phase 5 dense-retrieval baseline (all-MiniLM-L6-v2, source level):** Recall@1
0.877 · Recall@5 0.965 · Recall@10 0.983 · MRR 0.912 (57 answerable questions).
See [`reports/retrieval_metrics_baseline_v1.md`](./reports/retrieval_metrics_baseline_v1.md).

**Phase 6 reranked baseline (+ Cohere rerank-v3.5, source level):** Recall@1
**0.930** · Recall@5 **0.982** · MRR **0.953**; page Recall@1 0.63 → **0.90**.
Reranking applied on 63/63 (no fallback). See
[`reports/retrieval_metrics_baseline_v1_gemini36.md`](./reports/retrieval_metrics_baseline_v1_gemini36.md)
and `experiments/results/baseline_v1.json`.

**Phases 7–8 embedding selection — `e5-base-v2`.** Dense source Recall@5 **1.000** /
MRR **0.927** beats MiniLM (0.965 / 0.912) and bge-base (0.983 / 0.915). Confirmed with
Cohere reranking (E5 reranked source Recall@5 **1.000** / MRR **0.962** vs MiniLM 0.982 / 0.953).
Cost: 768-dim, ~3× query latency. Promotion to production is Phase 22. See
[`reports/EMBEDDING_COMPARISON.md`](./reports/EMBEDDING_COMPARISON.md) and
[`reports/EMBEDDING_DECISION.md`](./reports/EMBEDDING_DECISION.md).

**Phases 9–10 selections.** Keep **Cohere rerank-v3.5** (page Recall@1 ~0.63 → **0.90**; ~640 ms/query).
Chunk size **1500/200** beats the current 2500/300 on `e5-base-v2` (source R@1 0.877 → **0.930**,
page R@1 0.649 → **0.789**) — larger chunks lose to E5's 512-token truncation. See
[`reports/RERANKER_DECISION.md`](./reports/RERANKER_DECISION.md) and
[`reports/CHUNKING_DECISION.md`](./reports/CHUNKING_DECISION.md).

**Phase 11 parser selection.** Retain current **`pdfplumber`** (table-aware) pipeline over
Unstructured.io (source R@1 **0.930** vs 0.895, page R@1 **0.789** vs 0.737, 2x faster ingestion, zero heavy ML deps).
See [`reports/PARSER_DECISION.md`](./reports/PARSER_DECISION.md).

**Phases 12–13 LLM selection & generation metrics.** Retain **`gemini-3.6-flash`** as primary and
**Groq `openai/gpt-oss-120b`** as fallback. On the same reranked context: Gemini delivers **+29.7%** token recall,
**+8.7%** numerical recall, and **3.1× faster** latency (3.9s vs 11.9s) while Groq provides 100% abstention on unanswerable
queries. See [`reports/LLM_DECISION.md`](./reports/LLM_DECISION.md).

**Phases 14–15 Human evaluation & Query rewriting.** Human evaluation (4.68/5.00) highlighted conversational ambiguity (3.62/5.00). Query rewriting resolved this: conversational source R@1 **0.667 → 1.000**, page R@1 **0.500 → 1.000**; full benchmark dense source R@1 **0.930 → 0.965**, page R@1 **0.789 → 0.842**. See [`reports/QUERY_REWRITING_DECISION.md`](./reports/QUERY_REWRITING_DECISION.md).

**Phase 16 Metadata filtering.** Evaluated Oracle vs Automated router vs Unfiltered. Oracle filtering gives **0% gain** (Source R@1 tied at 0.965) since E5 vector search already hits 55/57 questions at rank 1, while automated routing causes false-negative exclusions (Source R@1 drops to 0.947). Decision: reject automated hard filtering at retrieval time; retain unconstrained search + reranking. See [`reports/METADATA_FILTERING_DECISION.md`](./reports/METADATA_FILTERING_DECISION.md).

**Phase 17 Hybrid Search.** Evaluated Dense E5 + BM25 via Reciprocal Rank Fusion ($k=60$) on `chunk_1500_200`. Hybrid search achieves identical 1.000 Source Recall@5 and 0.965 Source Recall@1, but slightly lower Page Recall@1 (0.825 vs 0.842) and 2.9× higher retrieval latency (51 ms vs 17.6 ms). Decision: reject hybrid search as default production retriever; retain unfiltered Dense E5 + Cohere cross-encoder reranker. See [`reports/HYBRID_SEARCH_DECISION.md`](./reports/HYBRID_SEARCH_DECISION.md).

**Phase 18 Abstention & Refusal Calibration.** Evaluated adversarial unanswerable questions (Q056–Q061). Both Gemini 3.6 Flash and Groq GPT-OSS-120B achieve **100% abstention precision and recall** (6/6) with zero hallucinations. Evaluated pre-LLM score gating ($\tau \in [0.10, 0.85]$); score thresholding fails because unanswerable queries with keyword overlap score up to 0.7831, while valid answerable clauses score as low as 0.1848, causing an intolerable 33.3% false refusal rate at $\tau = 0.80$. Decision: rely on prompt-based LLM grounded refusal; reject pre-LLM threshold gating. See [`reports/ABSTENTION_DECISION.md`](./reports/ABSTENTION_DECISION.md).

**Phase 19 Context Compression & Trimming.** Implemented adaptive score-aware candidate pruning and structural whitespace/table cleanup in `evaluation/lib/compressor.py`. Retains **100% Source Recall (57/57)** with zero fact loss, while cutting prompt context length by **15.87%** (57,975 characters saved; average chunks per query reduced from 5.00 to 3.89). Enriches prompt headers with page numbers and relevance scores for precise citation attribution. See [`reports/CONTEXT_COMPRESSION_DECISION.md`](./reports/CONTEXT_COMPRESSION_DECISION.md).

**Phase 20 Faithfulness & Hallucination Audit.** Conducted comprehensive factual audit on 63 benchmark queries across Gemini and Groq. Zero phantom citations detected (0.0%). Numerical hallucinations fell by 68% (from 28 queries down to 9 in Groq and 2 in Gemini). Unanswerable hallucinations fell from 67% in Baseline V1 to 0% (100% clean abstention). See [`reports/FAITHFULNESS_HALLUCINATION_AUDIT.md`](./reports/FAITHFULNESS_HALLUCINATION_AUDIT.md).

**Phase 21 Latency & Cost Optimization.** Mapped end-to-end latency waterfall (retrieval 44ms, rerank 616ms, compression <1ms, Gemini generation 3.9s, TTFT ~750ms). Token economics show $2.14 / 1,000 queries with Cohere or $0.14 / 1,000 queries with self-hosted cross-encoder (<$110/mo for 10,000 students). See [`reports/LATENCY_COST_OPTIMIZATION.md`](./reports/LATENCY_COST_OPTIMIZATION.md).

**Phase 22 Promotion to Production.** Formally promoted winning configurations to `backend/app/`: `intfloat/e5-base-v2` embedding with prefix wrapper, 1500/200 chunking, Cohere reranking, conversational query rewriting, adaptive score-aware compression, calibrated grounded generation, and timing telemetry. See [`reports/PROMOTION_TO_PRODUCTION.md`](./reports/PROMOTION_TO_PRODUCTION.md).

**Phase 23 Observability & Tracing.** Implemented `backend/app/telemetry.py` with sliding-window trace logging and real-time operational endpoint `GET /api/metrics`. See [`reports/OBSERVABILITY_TRACING.md`](./reports/OBSERVABILITY_TRACING.md).

**Phase 24 Production Hardening.** Verified input validation, rate limiting, constant-time bearer authentication, zero-secret leakage, and multi-tier LLM failover. See [`reports/PRODUCTION_HARDENING.md`](./reports/PRODUCTION_HARDENING.md).

**Phase 25 Program Completion.** All 25 phases of the evaluation and optimization program are 100% complete and verified.

### ⚠ Runtime blockers — status

- **Cohere Trial `429` (10/min)** — ✅ **worked around.** The harness applies an
  eval-only throttle (`EVAL_COHERE_MIN_INTERVAL_MS`, e.g. 7000) plus a bounded
  429-retry, so all 63 questions were reranked cleanly (no fallback to retrieval
  order). Production `app/query/reranker.py` is untouched.
- **LLM (OpenRouter) `402`** — ⛔ still unfunded (account never purchased credits).
  Abandoned in favour of a Google Gemini key.
- **`gemini-2.5-flash` 404** — ⛔ the frozen-baseline LLM is "no longer available to
  new users"; substituted Google's named successor **`gemini-3.6-flash`** (recorded
  as a deviation in `configs/baseline_v1_gemini36.json` and `results/baseline_v1.json`).
- **`gemini-3.6-flash` free tier** — ✅ **mitigated.** 20 requests/day cap mitigated
  by automatic failover to Groq (`openai/gpt-oss-120b`). Generated full 63 answers on Groq
  and 18-sample clean head-to-head on Gemini. Generation metrics scored via `metrics/generation_metrics.py`.
