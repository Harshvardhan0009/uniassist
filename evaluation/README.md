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
| **7** | **embedding comparison** (MiniLM vs BGE vs E5) | **done — `EMBEDDING_DECISION.md` recommends e5-base-v2** |

**Phase 5 dense-retrieval baseline (all-MiniLM-L6-v2, source level):** Recall@1
0.877 · Recall@5 0.965 · Recall@10 0.983 · MRR 0.912 (57 answerable questions).
See [`reports/retrieval_metrics_baseline_v1.md`](./reports/retrieval_metrics_baseline_v1.md).

**Phase 6 reranked baseline (+ Cohere rerank-v3.5, source level):** Recall@1
**0.930** · Recall@5 **0.982** · MRR **0.953**; page Recall@1 0.63 → **0.90**.
Reranking applied on 63/63 (no fallback). See
[`reports/retrieval_metrics_baseline_v1_gemini36.md`](./reports/retrieval_metrics_baseline_v1_gemini36.md)
and `experiments/results/baseline_v1.json`.

**Phase 7 embedding comparison (dense, source level):** `e5-base-v2` Recall@5
**1.000** / MRR **0.927** beats MiniLM (0.965 / 0.912) and bge-base (0.983 / 0.915),
and wins page-level too — at 768-dim and ~3× query latency. Recommended (promotion
gated on approval). See [`reports/EMBEDDING_COMPARISON.md`](./reports/EMBEDDING_COMPARISON.md)
and [`reports/EMBEDDING_DECISION.md`](./reports/EMBEDDING_DECISION.md).

### ⚠ Runtime blockers — status (updated Phase 6)

- **Cohere Trial `429` (10/min)** — ✅ **worked around.** The harness applies an
  eval-only throttle (`EVAL_COHERE_MIN_INTERVAL_MS`, e.g. 7000) plus a bounded
  429-retry, so all 63 questions were reranked cleanly (no fallback to retrieval
  order). Production `app/query/reranker.py` is untouched.
- **LLM (OpenRouter) `402`** — ⛔ still unfunded (account never purchased credits).
  Abandoned in favour of a Google Gemini key.
- **`gemini-2.5-flash` 404** — ⛔ the frozen-baseline LLM is "no longer available to
  new users"; substituted Google's named successor **`gemini-3.6-flash`** (recorded
  as a deviation in `configs/baseline_v1_gemini36.json` and `results/baseline_v1.json`).
- **`gemini-3.6-flash` free tier** — ⚠ **20 requests/day** cap. Only 16/63 answers
  were captured with a real LLM; the rest fell back to extractive text. Full answer
  capture (Phases 13-14) needs a paid tier or multi-day capture. Retrieval/rerank
  are complete and do not depend on the LLM.
