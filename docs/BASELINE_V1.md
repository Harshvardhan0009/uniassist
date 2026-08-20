# Baseline V1 — Frozen Reference Configuration (Phase 1)

> **This is the stable reference point for the entire evaluation program.**
> Every future experiment changes **exactly one** major variable relative to this baseline and is
> compared back to it. **Do not modify Baseline V1.** Machine-readable copy:
> [`evaluation/configs/baseline_v1.json`](../evaluation/configs/baseline_v1.json).

- **Frozen on:** 2026-08-20
- **Baseline code commit:** `7881175` (branch `laukik-uniassist-branch`)
- **Environment:** Python 3.13.1; exact package pins in [`backend/requirements.lock`](../backend/requirements.lock)
- **Status:** FROZEN

---

## 1. What "Baseline V1" is

Baseline V1 is the **current production configuration** exactly as deployed today — nothing was
replaced, retuned, or "improved" in this phase. It matches the target baseline in the master plan.

| Component | Baseline V1 value |
|---|---|
| **Parser** | `pdfplumber` (table-aware → Markdown tables) with `PyPDFLoader` fallback; `python-docx`; LangChain `TextLoader` |
| **Chunking** | `RecursiveCharacterTextSplitter`, **2500** chars / **300** overlap; keep-whole if ≤ 2500 |
| **Indexed content** | **LLM summary** as `page_content` (raw text preserved in `metadata.raw_content`) |
| **Summarizer** | `google/gemini-2.5-flash` (OpenRouter), temp **0.0**, max_tokens **256**, concurrency **5** |
| **Embedding** | **all-MiniLM-L6-v2**, 384-dim, normalized, cosine, in-process |
| **Vector store** | **ChromaDB**, collection `university_docs`, `hnsw:space = cosine` |
| **Retrieval** | top-**K = 20**, `min_relevance_score = 0.0` (disabled) |
| **Reranker** | **Cohere `rerank-v3.5`**, top-**N = 5** — **enabled** |
| **LLM (generation)** | **google/gemini-2.5-flash** (OpenRouter), temp **0.1**, max_tokens **1024**, ≤ 6 history msgs |
| **Corpus** | `Data/` — 14 ingestible files (12 PDF + 1 DOCX + 1 TXT); 1 image ignored |

> **Correction vs. `ARCHITECTURE.md`:** an earlier version of `ARCHITECTURE.md` stated Cohere was
> not configured. The live `.env` **does** provide a Cohere key, so **reranking is active** and is
> part of Baseline V1. `ARCHITECTURE.md` has been updated to reflect this.

---

## 2. Frozen parameters (authoritative)

These are the exact values a re-run must reproduce. Source of truth is
`evaluation/configs/baseline_v1.json`; this table mirrors it for humans.

```
parser.pdf              = pdfplumber (table-aware) + PyPDFLoader fallback
parser.docx             = python-docx
parser.txt_md           = langchain TextLoader (utf-8)
chunking.size           = 2500
chunking.overlap        = 300
chunking.keep_whole     = <= 2500 chars
indexing.content        = llm_summary  (raw in metadata.raw_content)
summarizer.temperature  = 0.0
summarizer.max_tokens   = 256
embedding.model         = all-MiniLM-L6-v2
embedding.dimensions    = 384
embedding.normalize     = true
embedding.similarity    = cosine
vector_store.engine     = chromadb
vector_store.collection = university_docs
vector_store.space      = cosine
retrieval.top_k         = 20
retrieval.min_score     = 0.0
reranker.model          = rerank-v3.5 (cohere)
reranker.top_n          = 5
reranker.enabled        = true
generation.model        = google/gemini-2.5-flash (openrouter)
generation.temperature  = 0.1
generation.max_tokens   = 1024
generation.max_history  = 6
```

---

## 3. Pinned environment (reproducibility)

`requirements.txt` uses loose `>=` bounds; the installed stack is well ahead of them. To make the
baseline reproducible, exact versions are frozen in **`backend/requirements.lock`**. Key pins:

| Package | Pinned |
|---|---|
| langchain / langchain-core | 1.3.15 / 1.5.4 |
| langchain-chroma / chromadb | 1.1.0 / 1.5.9 |
| langchain-openai / openai | 1.5.0 / 3.0.0 |
| langchain-cohere / cohere | 0.6.0 / 5.21.1 |
| langchain-huggingface / sentence-transformers | 1.2.2 / 5.7.0 |
| transformers / torch | 5.15.0 / 2.13.0 |
| pdfplumber / pypdf | 0.11.10 / 6.16.1 |
| scikit-learn / numpy / scipy | 1.9.0 / 2.5.2 / 1.18.0 |
| langgraph / langsmith | 1.2.11 / 0.10.18 (installed; unused so far) |

Reproduce the environment with:

```bash
python -m venv venv && . venv/Scripts/Activate.ps1   # Windows
pip install -r requirements.lock
```

> Not yet installed (added only when their phase begins): `ragas` (generation metrics),
> `rank-bm25` (hybrid search), `datasets`.

---

## 4. Freeze rules (apply to every experiment)

1. **One major variable at a time** — change only the component under test; hold everything else at
   Baseline V1.
2. **Same evaluation dataset** for every run (Phase 2).
3. **Same corpus** unless the experiment is specifically about parsing (Phase 11).
4. **Always record latency** (embedding / retrieval / reranking / generation / total).
5. **Never delete previous results** — every experiment stays reproducible under
   `evaluation/experiments/results/`.
6. **Never mutate production during an experiment** — use a **separate Chroma collection** per
   experiment (e.g. `eval_minilm_2500`, `eval_bge_2500`), ideally on a **local** Chroma to avoid
   remote cold-start latency and cost.

---

## 5. How Baseline V1 will be *measured*

Phase 1 only **defines and freezes** the configuration. The actual baseline **numbers**
(Recall@1/5/10/20, MRR, latency, later faithfulness/correctness) are produced in **Phase 6** and
written to `evaluation/experiments/results/baseline_v1.json`, once the evaluation dataset (Phase 2–3)
and pipeline (Phase 4–5) exist.

---

## 6. Open decisions to settle in Phase 4 (eval framework design)

These do not affect the freeze itself but must be decided before measuring:

1. **Index summaries (production-faithful) vs. raw text (deterministic).** Baseline V1 indexes
   summaries. Because summaries are non-deterministic, experiments should **snapshot one summarized
   chunk set** and re-embed that fixed set per embedding model — or switch to raw-text indexing for
   experiments. Choose one and hold it constant within a comparison.
2. **Experiment vector store:** local persistent Chroma (recommended) vs. the remote Render instance.
3. **Baseline dataset size:** start at 50 questions, target 100+ (Phase 2).
