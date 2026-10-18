# Phase 11 — Parser experiments

Embedding **`intfloat/e5-base-v2`** held constant; chunk size **1500 / 200** (the Phase 10 selection) held constant; same corpus (14 files, 144 pages), `top_k=20`, cosine, 57 answerable questions scored (6 unanswerable excluded). **Only the parser changes.** Both configurations use fresh frozen snapshots (raw text, no LLM summarisation).

- **Control:** Current parser (`pdfplumber` table-aware with Markdown conversion, `PyPDFLoader` fallback, `python-docx`, `TextLoader`)
- **Test:** `Unstructured.io` parser (`partition` with `strategy="fast"`, element typing, HTML-table to Markdown conversion)

---

## 1. Overall Metrics (Dense Retrieval & Reranked)

### Dense Retrieval (top-20, dense-only)

| Parser | #chunks | src R@1 | src R@5 | src R@10 | src MRR | page R@1 | page R@5 | page R@10 | page MRR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **Current (`pdfplumber`) ←best** | 285 | **0.930** | **1.000** | **1.000** | **0.952** | **0.789** | 0.860 | **0.930** | **0.819** |
| Unstructured (`fast`) | 231 | 0.895 | **1.000** | **1.000** | 0.932 | 0.737 | **0.877** | 0.895 | 0.792 |
| *Delta (Unstructured vs Current)* | *-54 (-19%)* | *-0.035* | *0.000* | *0.000* | *-0.020* | *-0.052* | *+0.017* | *-0.035* | *-0.027* |

### Reranked (top-5 Cohere `rerank-v3.5`)

| Parser | src R@1 | src R@5 | src MRR | page R@1 | page R@5 | page MRR | P@5 (src) | P@5 (page) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **Current (`pdfplumber`)** | 0.930 | 1.000 | 0.953 | **0.877** | **0.965** | 0.906 | **0.796** | **0.393** |
| Unstructured (`fast`) | **0.965** | 1.000 | **0.977** | **0.877** | **0.965** | **0.912** | 0.747 | 0.319 |

---

## 2. By Category (Dense Retrieval, Source Level)

| Category | n | Current R@1 | Unstructured R@1 | Current MRR | Unstructured MRR | Delta MRR |
|---|--:|--:|--:|--:|--:|--:|
| `academic` | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| `conversational` | 6 | **0.667** | 0.500 | **0.742** | 0.631 | -0.111 |
| `direct_factual` | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| `exact_terminology` | 7 | **0.857** | 0.714 | **0.905** | 0.833 | -0.072 |
| `multi_condition` | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| `placement` | 7 | 0.857 | 0.857 | 0.929 | 0.929 | 0.000 |
| `policy` | 9 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| `table_based` | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

---

## 3. By Difficulty (Dense Retrieval, Source Level)

| Difficulty | n | Current R@1 | Unstructured R@1 | Current MRR | Unstructured MRR |
|---|--:|--:|--:|--:|--:|
| `easy` | 15 | **0.867** | 0.800 | **0.906** | 0.869 |
| `medium` | 37 | **0.946** | 0.919 | **0.965** | 0.948 |
| `hard` | 5 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 4. Latency & Corpus Footprint

| Metric | Current (`pdfplumber`) | Unstructured (`fast`) |
|---|---|---|
| Ingestion Chunks | 285 chunks | 231 chunks (-19%) |
| Snapshot Extraction Time | 10.79s | 21.27s (2x slower) |
| Dense Retrieval Latency (avg) | 52.35 ms | 47.78 ms |
| Dense Retrieval Latency (p95) | 72.39 ms | 58.14 ms |
| Reranker Latency (avg) | 616.69 ms | 827.11 ms |
