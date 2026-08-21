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
| 5 | `metrics/retrieval_metrics.py` (Recall@1/5/10/20, MRR, Precision@K, HitRate@K) | pending |
| 6 | official `results/baseline_v1.json` reference numbers | pending (needs working keys) |

### ⚠ Runtime blockers for the *faithful* baseline (discovered in Phase 4)

The pipeline is complete and runs, but two **external API keys** are currently
non-functional, which blocks a production-faithful Baseline V1 (summaries +
Cohere rerank + LLM answers):

1. **LLM (OpenRouter) → `402 Payment Required`.** No credit, so chunk
   summarisation (indexing) and answer generation are unavailable. The current
   snapshot therefore freezes **raw chunk text** (`summarizer.effective = false`),
   and full-RAG answers fall back to extractive text (`has_llm = false`).
2. **Cohere → Trial key, 10 calls/min (`429`).** Reranking is rate-limited across
   the 63-question benchmark and falls back to retrieval order for most items.

Retrieval (MiniLM, local) is fully functional and reproducible. Resolving the two
keys (or an explicit decision to index raw text / skip reranking) is required
before the Phase 6 baseline numbers can be considered faithful.
