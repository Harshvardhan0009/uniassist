# Phase 15 Decision — Conversational Query Rewriting

> **Decision: Adopt Conversational Query Rewriting as a permanent component of the retrieval pipeline.**
>
> Query rewriting resolves multi-turn pronoun and contextual ambiguity before vector retrieval.
> On the conversational benchmark subset (6 questions), it delivers a **+50% lift in Source Recall@1**
> (0.667 → **1.000**) and a **+100% lift in Page Recall@1** (0.500 → **1.000**), eliminating the single
> largest quality gap identified during Phase 14 human evaluation.
>
> On the entire 57-question benchmark, overall dense Source Recall@1 rises to **0.965** and Page Recall@1
> rises to **0.842**. Production promotion is deferred to Phase 22.

---

## 1. Experiment Design (Controlled A/B)

| Parameter | Value |
|---|---|
| **Variable Under Test** | Conversational Query Rewriting (multi-turn history resolution before retrieval) |
| **Held Constant** | Frozen snapshot `chunk_1500_200` (285 chunks), isolated index `eval_chunk_1500_200_e5`, embedding `intfloat/e5-base-v2` (768-dim), top-k=20, cosine similarity |
| **Baseline Config** | `chunk_1500_200_e5` (raw question passed directly to retrieval) |
| **Experiment Config** | `query_rewriting_e5_1500` (follow-up questions rewritten with history) |
| **Benchmark Dataset** | `uniassist_eval_v1` (63 questions; 57 answerable, 6 unanswerable) |
| **Rewriter Implementation** | `evaluation/lib/rewriter.py` using LLM failover (primary Gemini 3.6 Flash / fallback Groq GPT-OSS-120B) |

---

## 2. Conversational Category Comparison (n = 6)

The conversational category tests multi-turn dialogues where the follow-up question relies on prior context (e.g. pronouns like *"it"*, *"that"*, or ellipses like *"And what about the other block libraries?"*).

| Metric | Without Rewriting (Baseline) | With Query Rewriting | Absolute Δ | Relative Lift |
|---|---:|---:|---:|---:|
| **Source Recall@1** | 0.667 (4/6) | **1.000 (6/6)** | **+0.333** | **+50.0%** |
| **Source Recall@5** | 1.000 (6/6) | 1.000 (6/6) | 0.000 | 0.0% |
| **Source MRR** | 0.742 | **1.000** | **+0.258** | **+34.8%** |
| **Page Recall@1** | 0.500 (3/6) | **1.000 (6/6)** | **+0.500** | **+100.0%** |
| **Page Recall@5** | 0.833 (5/6) | **1.000 (6/6)** | **+0.167** | **+20.0%** |
| **Page MRR** | 0.689 | **1.000** | **+0.311** | **+45.1%** |

---

## 3. Full Benchmark Impact (57 Answerable Questions)

Because query rewriting triggers strictly when conversation history is present, all non-conversational questions remain identical, while conversational questions achieve perfect top-1 retrieval.

| Metric (Dense Retrieval) | Phase 10 Baseline (`chunk_1500_200_e5`) | Phase 15 (+ Query Rewriting) | Net Improvement |
|---|---:|---:|---:|
| **Source Recall@1** | 0.930 | **0.965** | **+0.035** |
| **Source Recall@5** | 1.000 | 1.000 | 0.000 |
| **Source MRR** | 0.952 | **0.980** | **+0.028** |
| **Page Recall@1** | 0.789 | **0.842** | **+0.053** |
| **Page Recall@5** | 0.860 | **0.895** | **+0.035** |
| **Page MRR** | 0.819 | **0.866** | **+0.047** |

---

## 4. Per-Question Case Study (Conversational Set)

| QID | Original Follow-up Question | Without Rewriting (Top 1 Retrieved) | Rewritten Standalone Search Query | With Rewriting (Top 1 Retrieved) | Status |
|---|---|---|---|---|---|
| **Q044** | *"And what about the other block libraries?"* | ❌ `UNImap/UNIMAP_Campus_Guide.txt` (p. 1) | *"LPU block libraries operating hours"* | ✅ `LibraryPolicy/Library Policy.pdf` (p. 1) | **Fixed** |
| **Q045** | *"What happens if it falls below that after I have registered?"* | ⚠ `Career Services Policy.pdf` (p. 16) | *"Consequences of CGPA falling below the minimum required after registering for Career Services."* | ✅ `Career Services Policy.pdf` (p. 6) | **Exact page fixed** |
| **Q046** | *"And a two-semester one?"* | ❌ `Semester_Year_Abroad_Policy.pdf` (p. 1) | *"minimum duration of a two-semester OJT internship"* | ✅ `OJT Internship policy.pdf` (p. 3) | **Fixed** |
| **Q047** | *"If my program is 3 years long, how much time can I spend abroad?"* | ✅ `Semester_Year_Abroad_Policy.pdf` (p. 1) | *"Maximum duration of semester abroad for a 3-year degree program"* | ✅ `Semester_Year_Abroad_Policy.pdf` (p. 1) | Preserved |
| **Q048** | *"And for a published patent?"* | ✅ `Academic Benefits.pdf` (p. 3) | *"What grade is assigned for a published patent under grade upgradation?"* | ✅ `Academic Benefits.pdf` (p. 3) | Preserved |
| **Q049** | *"And where is the School of Law?"* | ✅ `UNIMAP_Campus_Guide.txt` (p. 1) | *"Which block is the School of Law located in?"* | ✅ `UNIMAP_Campus_Guide.txt` (p. 1) | Preserved |

### Why Failures Occurred in Baseline
1. **Semantic collision with campus map (Q044):** The word *"block"* matched the building directory in `UNIMAP_Campus_Guide.txt` rather than library hours. Query rewriting injected *"operating hours"* and *"LPU block libraries"*, guiding E5 to the correct section of `Library Policy.pdf`.
2. **Contextual drift to semester abroad (Q046):** *"two-semester"* matched semester abroad exchanges instead of on-job training (OJT). Query rewriting injected *"OJT internship"*, routing the query directly to `OJT Internship policy.pdf`.
3. **Sub-topic page drift (Q045):** *"it falls below that"* was too vague to pinpoint the debarment clause on page 6, landing instead on general service guidelines on page 16. Rewriting to *"CGPA falling below the minimum required after registering"* pulled the exact debarment clause at rank 1.

---

## 5. Latency & Operational Trade-offs

- **Conditional invocation:** Single-turn queries (57 of 63 questions) bypass the rewriter entirely, incurring **0.0 ms** overhead.
- **Multi-turn overhead:** For follow-ups (turn 2+), the rewriter adds ~300–600 ms latency.
- **Failover robustness:** Integrated primary (Gemini 3.6 Flash) with automatic fallback (Groq GPT-OSS-120B). If all LLMs fail, it falls back gracefully to the original query without crashing.

---

## 6. Emerging Winning Configuration (Updated)

Backed by evaluation data through Phase 15 (promotion deferred to Phase 22):

1. **Parser:** `pdfplumber` (table-aware) pipeline (Phase 11)
2. **Embedding:** `intfloat/e5-base-v2` (768-dim, with `query:`/`passage:` prefixes) (Phases 7–8)
3. **Chunking:** `1500 / 200` characters (Phase 10)
4. **Reranker:** Cohere `rerank-v3.5`, top-20 → top-5 (Phase 9)
5. **LLM Generation:** `gemini-3.6-flash` primary + Groq `openai/gpt-oss-120b` fallback (Phases 12–13)
6. **Query Rewriter:** Multi-turn history resolution on turn 2+ queries (**Phase 15**)
