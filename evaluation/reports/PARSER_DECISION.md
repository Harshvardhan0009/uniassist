# Parser decision (Phase 11)

**Date:** 2026-10-06 · **Status:** ✅ **RETAIN CURRENT PARSER (`pdfplumber` / table-aware)**. Promotion of winning pipeline components remains deferred to **Phase 22**. · **Dataset:** `uniassist_eval_v1` (57 answerable, 6 unanswerable) · **Method:** dense retrieval + Cohere rerank-v3.5, `intfloat/e5-base-v2` and `1500/200` chunking held constant, raw text (no LLM summarisation).

Full metrics breakdown: [`PARSER_COMPARISON.md`](./PARSER_COMPARISON.md).

---

## 1. Summary of Results

| Metric | Current (`pdfplumber`) | Unstructured (`fast`) | Delta | Winner |
|---|---|---|---|:---:|
| **Source Recall@1 (dense)** | **0.930 (53/57)** | 0.895 (51/57) | -0.035 (-2 Qs) | **Current** |
| **Source MRR (dense)** | **0.952** | 0.932 | -0.020 | **Current** |
| **Page Recall@1 (dense)** | **0.789 (45/57)** | 0.737 (42/57) | -0.052 (-3 Qs) | **Current** |
| **Page MRR (dense)** | **0.819** | 0.792 | -0.027 | **Current** |
| **Page Recall@1 (reranked)** | 0.877 (50/57) | 0.877 (50/57) | 0.000 | *Tie* |
| **Page Recall@5 (reranked)** | 0.965 (55/57) | 0.965 (55/57) | 0.000 | *Tie* |
| **Ingestion Time** | **10.79s** | 21.27s | +10.48s (2x) | **Current** |
| **Heavy Dependencies** | **None (lightweight)** | Torch, OpenCV, ONNX, Timm | +500MB+ | **Current** |

> *Note on sample size: Metric deltas between parsers represent shifts of 2–3 questions (out of 57 answerable) on the dense retrieval stage; decision is further reinforced by ingestion speed (2× faster) and avoiding heavy PyTorch/OCR dependencies.*

---

## 2. Key Findings

1. **Dense Retrieval Quality Favors Current Parser:**
   - The current parser achieves higher first-stage accuracy at both the source level (Recall@1: **0.930** vs 0.895) and the critical page level (Recall@1: **0.789** vs 0.737).
   - In exact-terminology questions, the current parser scores MRR **0.905** vs 0.833 for Unstructured.
   - In conversational queries, the current parser scores MRR **0.742** vs 0.631 for Unstructured.
   - Table, policy, and academic categories perform at 1.000 under both parsers.

2. **Reranked Equivalence:**
   - With Cohere `rerank-v3.5` enabled, page-level performance is identical (Page Recall@1: **0.877**, Page Recall@5: **0.965**). Unstructured provides a slight bump in Source Recall@1 (0.965 vs 0.930), but no gain on fine-grained page citation precision.

3. **Operational Complexity & Dependency Burden:**
   - `pdfplumber` relies only on standard Python libraries (`pdfminer.six`, `pypdf`, `python-docx`), running stably on Windows with Python 3.13.
   - `unstructured` introduces transitive dependencies on PyTorch, Torchvision, OpenCV, ONNX, Timm, Accelerate, and Matplotlib even for its `fast` strategy, due to unconditional top-level imports in its internal PDF utilities.
   - Furthermore, `unstructured[pdf]` officially excludes `unstructured-inference` on Windows for Python >= 3.13, creating deployment and packaging fragility.

---

## 3. Decision & Recommendation

**Retain the current parser (`pdfplumber` table-aware pipeline). Do NOT switch to Unstructured.io.**

- The current parser delivers superior dense retrieval accuracy, runs twice as fast during ingestion (10.8s vs 21.3s), and maintains a clean, lightweight dependency footprint.
- Production code under `app/ingestion/` remains unchanged and stable.
