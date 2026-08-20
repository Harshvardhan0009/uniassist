# UniAssist / LPUAssist — Current State Audit (Phase 0)

> **Purpose:** A factual, code-verified snapshot of the system **as it exists today**, before any
> evaluation or improvement work begins. This is the reference for Phase 1 (Freeze Baseline).
> Everything here was verified by reading the source, `requirements.txt`, the live `backend/.env`
> (secrets redacted), installed package versions, and the `Data/` corpus on
> **2026-08-20**. No production behavior was changed while producing this document.

---

## 1. Architecture (as implemented)

Two independently deployable parts:

- **Backend** — FastAPI service exposing an offline **ingestion** pipeline and an online **query**
  pipeline. Located in `backend/app/`.
- **Frontend** — Next.js App Router single-page chat UI (`frontend/app/page.js`).

### 1.1 Ingestion pipeline (offline CLI: `python -m app.ingestion.pipeline`)

```
Data/ (recursive)
  → partition_directory   (partition.py)   PDF/DOCX/TXT/MD → LangChain Documents (per page for PDFs)
  → chunk_documents       (chunker.py)     RecursiveCharacterTextSplitter, keep-whole if ≤ 2500 chars
  → summarize_chunks      (summarizer.py)  LLM summary becomes page_content; raw text → metadata.raw_content
  → embed_and_store       (embedder.py)    sanitize meta → delete existing by source_file → batch add (50)
  → ChromaDB              (collection: university_docs, hnsw:space = cosine)
```

### 1.2 Query pipeline (online: `app/query/chain.py::query`)

```
question + history
  → retrieve   (retriever.py)   similarity_search_with_relevance_scores, top_k = 20, drop < MIN_SCORE
  → rerank     (reranker.py)    Cohere rerank-v3.5 → top_n = 5 (skipped if no key; fallback to retrieval order)
  → generate   (generator.py)   context built from metadata.raw_content, temp 0.1, max_tokens 1024
  → answer + sources + has_llm + timing + candidate counts   (chain.py)
```

**Key architectural fact:** when an LLM key is configured, retrieval is performed against
**LLM-generated summaries** (`page_content`), while final answer generation uses the **raw** chunk
text stored in `metadata.raw_content`. Without an LLM key, raw text is used for both.

---

## 2. Models & external services (actually configured)

Verified from the live `backend/.env` (secrets redacted) and code defaults.

| Role | Configured value | Notes |
|---|---|---|
| **Embedding** | `all-MiniLM-L6-v2` | 384-dim, `normalize_embeddings=True`, cosine. In-process; CUDA auto-detected else CPU (`embedder.py::_select_device`). |
| **LLM (summarize + generate)** | OpenRouter `google/gemini-2.5-flash` | Set via **legacy `GROK_*`** env names (`GROK_API_KEY`/`GROK_BASE_URL=https://openrouter.ai/api/v1`/`GROK_MODEL`). Code defaults if unset: xAI `grok-3-mini` @ `https://api.x.ai/v1`. |
| **Reranker** | **Cohere `rerank-v3.5` — CONFIGURED** (key present in `.env`) | Reranking is **active** in the current deployment. |
| **Vector store** | ChromaDB **client-server** @ `uniassist-chroma.onrender.com:443` (SSL), collection `university_docs` | No local `backend/chroma_db/` directory exists — remote mode only. |

### Generation / retrieval parameters

| Parameter | Value | Location |
|---|---|---|
| Chunk size / overlap | `2500` / `300` chars | `chunker.py` (hard-coded constants) |
| Splitter | `RecursiveCharacterTextSplitter`, separators `["\n\n","\n",". "," ",""]` | `chunker.py` |
| Keep-whole threshold | ≤ `2500` chars → not split | `chunker.py` |
| Retrieval top-K | `20` | `RETRIEVAL_TOP_K` |
| Rerank top-N | `5` | `RERANK_TOP_N` |
| Min relevance score | `0.0` (disabled) | `RETRIEVAL_MIN_SCORE` |
| Summarize temp / max tokens | `0.0` / `256` | `summarizer.py` |
| Summarize concurrency | `5` | `SUMMARY_CONCURRENCY` |
| Generate temp / max tokens | `0.1` / `1024` | `generator.py` |
| Max history messages | `6` (frontend sends last 8 non-error turns) | `generator.py::MAX_HISTORY_MESSAGES` |

---

## 3. Data flow details (per module)

- **`partition.py`** — recurses `Data/` for `.pdf/.docx/.txt/.md` (skips dotfiles; images ignored).
  - PDF: `pdfplumber` page-by-page; tables → Markdown tables appended under `### Extracted Table Data:`.
    Falls back to `PyPDFLoader` on any exception. One Document **per page**; `category` =
    `TableAndText` if tables present else `NarrativeText`.
  - DOCX: `python-docx`; paragraphs + table rows flattened; single Document, `page_number = 1`.
  - TXT/MD: LangChain `TextLoader` (UTF-8); single Document, `page_number = 1`.
  - Each file keyed by **path relative to `Data/`** and written to `metadata.source_file` (prevents
    same-name collisions across folders).
- **`chunker.py`** — per-file; Documents ≤ 2500 chars kept whole, else split. Enriches metadata:
  `source_file`, `chunk_index` (0..N per file), `chunk_type` (`table`/`text_only` from `category`),
  `page_number`.
- **`summarizer.py`** — if LLM key set: concurrent `chain.batch` (bounded by `SUMMARY_CONCURRENCY`),
  `return_exceptions=True`; per-chunk failures fall back to raw text. Result Document has
  `page_content = summary`, `metadata.raw_content = original text`. No key → `_passthrough`
  (raw text as both `page_content` and `raw_content`).
- **`embedder.py`** — lazily builds a **singleton** embedding fn and vector store. `_sanitize_metadata`
  coerces non-primitive values (lists→`str(...)`, None→`""`). `_delete_existing_sources` removes
  prior vectors for the same `source_file` via `$in` (idempotent re-ingest). Batch add of 50 with
  deterministic IDs `"{source_file}__chunk_{chunk_index}"`. `check_chroma()` does a heartbeat
  (server) or dir check (local) and reports count **only if the store singleton is already built**.
- **`retriever.py`** — `similarity_search_with_relevance_scores(query, k=top_k)`; drops results below
  `RETRIEVAL_MIN_SCORE`; writes `relevance_score` (rounded, 4dp) into each doc's metadata.
- **`reranker.py`** — `CohereRerank(model, top_n).compress_documents(...)`. No key → returns
  `documents[:top_n]` (retrieval order). Any error → same fallback.
- **`generator.py`** — context from `_extract_raw` (`metadata.raw_content`, JSON-list aware) with
  `[Source i: <source_file>]` headers; injects up to 6 prior turns; `ChatOpenAI` temp 0.1,
  max_tokens 1024; strips `<think>…</think>` (and unclosed variants). No key → `_build_clean_answer`
  returns a structured extract (`has_llm=False`). `sources` = de-duplicated `source_file` list.
- **`chain.py`** — orchestrates retrieve→rerank→generate, records per-step + total timings, returns
  `candidates_retrieved` / `candidates_reranked`. Empty retrieval → canned "couldn't find" message.

---

## 4. API surface (`app/main.py`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | none | Config flags + **live Chroma connectivity** + indexed doc count (nullable). |
| `POST` | `/api/query` | optional per-IP rate limit | Ask a question (+ optional `history`). |
| `POST` | `/api/ingest` | **Bearer `INGEST_TOKEN`** | Run ingestion; **disabled unless `INGEST_TOKEN` set**. |

**`POST /api/query` request:** `{ question: str(3..1000), history: ChatMessage[<=20] }` where
`ChatMessage = { role: "user"|"assistant", content: str(<=8000) }`.
**Response:** `{ answer, sources[], has_llm, timing{retrieval,reranking,generation,total},
candidates_retrieved, candidates_reranked }`.

**`GET /api/health` response:** `{ status, llm_configured, cohere_configured, embedding_model,
collection, chroma_mode, chroma_host?, chroma_connected, documents_indexed? }`.

Other: CORS from `ALLOWED_ORIGINS`; best-effort in-memory per-IP rate limiter (per process); Swagger
at `/docs`, ReDoc at `/redoc`.

---

## 5. Environment variables (`config.py` / `.env`)

| Variable (canonical) | Aliases | Default | Currently set to |
|---|---|---|---|
| `ALLOWED_ORIGINS` | — | `http://localhost:3000,http://127.0.0.1:3000` | (default) |
| `INGEST_TOKEN` | — | `""` (ingest disabled) | not present in `.env` → disabled |
| `RATE_LIMIT_PER_MINUTE` | — | `0` (off) | (default) |
| `LLM_API_KEY` | `GROK_API_KEY` | `""` | set (redacted) via `GROK_API_KEY` |
| `LLM_BASE_URL` | `GROK_BASE_URL` | `https://api.x.ai/v1` | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | `GROK_MODEL` | `grok-3-mini` | `google/gemini-2.5-flash` |
| `COHERE_API_KEY` | — | `""` | **set (redacted)** |
| `COHERE_RERANK_MODEL` | — | `rerank-v3.5` | `rerank-v3.5` |
| `CHROMA_HOST` | — | `""` (local mode) | `uniassist-chroma.onrender.com` |
| `CHROMA_PORT` | — | `8001` | `443` |
| `CHROMA_SSL` | — | `false` | `true` |
| `CHROMA_AUTH_TOKEN` | — | `""` | empty |
| `CHROMA_COLLECTION` | — | `university_docs` | `university_docs` |
| `EMBEDDING_MODEL` | — | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |
| `SUMMARY_CONCURRENCY` | — | `5` | (default) |
| `RETRIEVAL_TOP_K` | — | `20` | `20` |
| `RERANK_TOP_N` | — | `5` | `5` |
| `RETRIEVAL_MIN_SCORE` | — | `0.0` | (default) |

Frontend: `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`).

---

## 6. Data corpus (`Data/`)

15 files total; **14 ingestible** (the `.jpg` is skipped). Sizes are on-disk bytes, not token counts.

| Category (folder) | File | Type |
|---|---|---|
| DressCode | Dress Code and Uniform Policy for Students.pdf | PDF |
| DressCode | FAQ's.pdf | PDF |
| EDU REV | Academic Benefits.pdf | PDF |
| LibraryPolicy | Library Policy.pdf | PDF |
| NSSPolicy | LPUNSSPOLICY.pdf | PDF |
| PlacementPloicy | 2023 Career Services Policy.pdf | PDF |
| PlacementPloicy | Academic Benefit Plan - 3rd Party Industry Tests - 2023 Batch.docx | **DOCX** |
| PlacementPloicy | Academic Benefit Plan - Clearing Competitive Exams.pdf | PDF |
| PlacementPloicy | Academic Benefit Plan - Placement and Internship - 2023 and 2024 Batch.pdf | PDF |
| PlacementPloicy | OJT Internship policy 2019-2020 onwards (10 Aug 2019).pdf | PDF |
| ResidentialFacilities | Charges For Residential Facilities.pdf | PDF |
| ResidentialFacilities | Residential Facilities Refund Guidelines.pdf | PDF |
| SemesterExchange | Semester_Year_Abroad_Policy.pdf | PDF |
| UNImap | UNIMAP_Campus_Guide.txt | TXT |
| UNImap | UNIMAP.jpg | JPG *(not ingested)* |

Breakdown: **12 PDF + 1 DOCX + 1 TXT** ingestible; folder name `PlacementPloicy` is a (harmless)
typo in the source tree. The exact **indexed chunk count** is not recorded here because it depends on
a live re-ingest (see §9).

---

## 7. Tech stack & installed versions

`requirements.txt` uses **loose lower-bound pins** (`>=`), but the **installed environment is far
ahead** of those minimums. Verified with `pip freeze` in `backend/venv` (Python **3.13.1**):

| Package | requirements.txt (min) | Installed |
|---|---|---|
| langchain | `>=0.3.25` | **1.3.15** |
| langchain-core | `>=0.3.59` | **1.5.4** |
| langchain-community | `>=0.3.24` | **0.4.2** |
| langchain-huggingface | `>=0.2.0` | 1.2.2 |
| langchain-chroma | `>=0.2.4` | 1.1.0 |
| langchain-openai | `>=0.3.18` | 1.5.0 |
| langchain-cohere | `>=0.4.3` | 0.6.0 |
| langchain-text-splitters | `>=0.3.0` | 1.1.2 |
| chromadb | `>=0.6.0` | 1.5.9 |
| sentence-transformers | `>=4.0.0` | 5.7.0 |
| pdfplumber | `>=0.11.0` | 0.11.10 |
| pypdf | `>=5.0.0` | 6.16.1 |
| fastapi | `>=0.115.0` | 0.141.1 |
| pydantic-settings | `>=2.9.0` | 2.15.0 |

Also present (not in `requirements.txt`): `torch 2.13.0`, `transformers 5.15.0`, `openai 3.0.0`,
`numpy 2.5.2`, **`scikit-learn 1.9.0`** (useful for metrics/cosine). **Not installed:** `rank-bm25`,
`datasets`, `ragas`, `langsmith` (needed only for later phases: hybrid search, RAGAS-style eval,
tracing).

Frontend: Next.js 16 (App Router, Turbopack), React 19, JavaScript (no TS), CSS Modules,
`localStorage` persistence. Citations render as **source basenames only** (`page.js` ~L275–280):
`📄 <filename-without-extension>` — **no page numbers** shown today (relevant to Phase 20).

Infra: `docker-compose.yml` runs `chromadb/chroma:latest` (host `8001` → container `8000`); current
deployment instead uses the hosted Render Chroma.

---

## 8. Existing tests & CI

- **No test files exist** anywhere in the repo (verified by glob for `test*/`, `*_test`, `*.test.*`
  across `.py/.js/.jsx/.ts`). No `pytest`/`jest` config, no CI workflow.
- Git: branch `laukik-uniassist-branch`, working tree clean; latest commit `7881175` (Windows UTF-8
  fix). `ARCHITECTURE.md` was added in `0c1b170`.

---

## 9. Assumptions the system currently relies on

1. **Retrieval is against summaries, not raw text** (when an LLM key is set). The embedding model
   embeds LLM summaries; answers are generated from `raw_content`. This is central to any
   embedding/chunking experiment design.
2. **Summaries are non-deterministic.** Even at `temperature=0.0`, provider-side variation means a
   re-ingest can change the indexed text — a **reproducibility hazard** for experiments (see §11).
3. **`source_file` + `page_number` metadata** are present on every chunk (PDF pages carry true page
   numbers; DOCX/TXT are `page_number = 1`). This is sufficient to evaluate source-level and
   page-level retrieval recall.
4. **`relevance_score`** is whatever `langchain-chroma`'s `similarity_search_with_relevance_scores`
   returns for cosine space (higher = more similar; commonly `1 − distance`). The exact scaling
   should be **verified empirically** before relying on thresholds/abstention.
5. **Single shared collection** (`university_docs`) on a **remote Render** instance. There is no
   isolation between experiments today; A/B work will need separate collections (see §11).
6. **Abstention** for out-of-corpus questions relies **only** on the system prompt (+ the empty-
   retrieval canned message). There is no score-based or validation-based guard.
7. **Cohere reranker order/scores are not surfaced** in the API response (only counts + `sources`).

---

## 10. Current limitations

- **No evaluation framework, no benchmark dataset, no tests, no tracing.**
- **No hybrid (dense+sparse) search**, **no metadata filtering**, **no query rewriting/routing**,
  **no document versioning** — all are roadmap items in `ARCHITECTURE.md`/`uniassist.md`.
- **Abstention is prompt-only** → hallucination risk on unanswerable questions is untested.
- **`/api/health` doc count is null** until the vector-store singleton is initialized by a first
  query/ingest (it does not build the store just to count).
- **Remote Render Chroma cold-starts** — first request after idle can add tens of seconds of
  latency, which will contaminate latency measurements unless warmed first.
- **Citations lack page numbers / supporting excerpts** in the UI.
- **`RETRIEVAL_MIN_SCORE = 0.0`** keeps even very weak matches in the candidate set.

---

## 11. Discrepancies & risks found during the audit

> Per the workflow, these are **flagged, not changed**. Each includes the recommended handling for
> later phases.

1. **Docs vs. reality — Cohere.** `ARCHITECTURE.md` §4 states Cohere is "not currently configured"
   and reranking is "skipped", but the live `.env` **does** contain a `COHERE_API_KEY`. **Reranking
   is active today.** → Treat Cohere rerank-v3.5 as part of Baseline V1 (matches the plan). Update
   `ARCHITECTURE.md` when docs are refreshed (Phase 25).
2. **Dependency drift / weak pinning (reproducibility risk).** `requirements.txt` uses `>=` bounds
   while the installed stack is a **major version ahead** (e.g. `langchain 1.3.15` vs `>=0.3.25`,
   `langchain-openai 1.5.0` vs `>=0.3.18`, `chromadb 1.5.9`, `openai 3.0.0`, `transformers 5.15.0`,
   `torch 2.13.0`). Reproducible experiments require an **exact-pinned lockfile**. → Recommend
   freezing exact versions (e.g. a `requirements.lock`) before Phase 6 so baseline numbers are
   reproducible. **No change made now.**
3. **Experiment isolation.** All work currently points at the single production collection
   `university_docs` on a shared remote server. Running embedding/chunking experiments here would
   overwrite production vectors. → The eval framework must use **separate collections per experiment**
   (e.g. `eval_minilm_2500`, `eval_bge_2500`) and, ideally, a **local** Chroma for experiments to
   avoid remote cold-start noise and cost. (Satisfies plan Rule 6.)
4. **Summary-indexing confounds embedding/chunking experiments.** Because `page_content` is an LLM
   summary, naive re-ingestion per embedding model would also re-roll summaries, violating "one
   variable at a time." → The eval pipeline should **snapshot the summarized/chunked Documents once**
   and re-embed that fixed set for each embedding model. Also decide explicitly whether experiments
   index **summaries** (production-faithful) or **raw text** (simpler/deterministic) — and keep it
   constant across a comparison.
5. **Legacy env naming.** The deployment configures an OpenRouter/Gemini model through `GROK_*`
   variables. Functionally fine (via `AliasChoices`) but confusing. → Cosmetic; migrate to `LLM_*`
   during a later config pass.
6. **Health count laziness.** `documents_indexed` can report `null` on a cold process. → Minor;
   optionally build/count lazily in a future hardening pass.

None of the above are blocking for producing a baseline; items **2, 3, 4** must be **respected in
the eval framework design** (Phase 4) to keep experiments valid and reproducible.

---

## 12. What Phase 0 did *not* do

- Did not run a live query, ingest, or connect to the remote Chroma (avoided touching production and
  incurring cold-start/cost). Live connectivity + exact indexed chunk count will be captured when the
  **baseline is measured** (Phases 1 & 6).
- Did not modify any code, config, dependency, or data.
