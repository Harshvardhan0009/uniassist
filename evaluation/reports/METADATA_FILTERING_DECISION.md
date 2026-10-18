# Phase 16 Decision — Metadata Filtering & Domain Routing

> **Decision: REJECT hard metadata filtering for the default query pipeline. Retain unconstrained retrieval + cross-encoder reranking.**
>
> Empirical evaluation across the 57 answerable benchmark questions proves that hard metadata filtering offers
> **0.0% retrieval gain** even under ideal oracle conditions (Source Recall@1 remains identical at 0.965),
> while real-world automated intent routing **degrades retrieval performance** (Source Recall@1 drops from
> **0.965 → 0.947**; Source Recall@5 drops from **1.000 → 0.982**) due to cross-domain policy overlap.
>
> Metadata should be preserved strictly for UI citation badges and explicit user-selected facet filters,
> NOT for automated pre-retrieval query constraints.

---

## 1. Experiment Design (Controlled A/B)

We evaluated three retrieval paradigms over the identical frozen corpus (`chunk_1500_200`, 285 chunks) and embedding model (`intfloat/e5-base-v2`, 768-dim) across all 57 answerable benchmark questions:

1. **Unfiltered Dense Retrieval (Phase 15 Baseline):**
   Full semantic vector search across all 285 chunks; no domain restriction.
2. **Oracle Metadata Filtering (Theoretical Upper Bound):**
   Retrieval candidate set strictly constrained to the ground-truth topic (`q["topic"]`), simulating a 100% perfect human router.
3. **Automated Metadata Filtering (Realistic Production Pipeline):**
   An LLM intent router classifies the user query into 1 of 9 domain topics prior to retrieval, applying a hard filter (`where={"domain": pred_domain}`).

---

## 2. Benchmark Comparison (57 Answerable Questions)

| Retrieval Paradigm | Router Accuracy | Source Recall@1 | Source Recall@5 | Page Recall@1 | Page Recall@5 | Net Impact vs Baseline |
|---|---:|---:|---:|---:|---:|---|
| **Unfiltered (Current Winning Stack)** | N/A | **0.965** | **1.000** | **0.842** | **0.895** | **Baseline (Optimal)** |
| **Oracle Filtered (Ideal Router)** | 100.0% | **0.965** | **1.000** | **0.842** | **0.895** | **±0.000 (No gain)** |
| **Automated Filtered (Real-World LLM Router)** | 96.5% | 0.947 | 0.982 | 0.825 | 0.877 | **-0.018 (Degraded)** |

---

## 3. Why Oracle Filtering Provides Zero Gain

On 55 out of 57 questions, the winning retrieval stack (`intfloat/e5-base-v2` with `1500/200` chunking and conversational query rewriting) **already retrieves the correct document at Rank 1 without any filtering**.

For the only 2 questions in the entire benchmark that missed Rank 1 in dense retrieval:
- **Q030 (Placement Category):**
  - Expected: `PlacementPloicy/OJT Internship policy 2019-2020 onwards.pdf`
  - Top 1 Retrieved: `PlacementPloicy/2023 Career Services Policy.pdf`
- **Q050 (Exact Terminology Category):**
  - Expected: `PlacementPloicy/2023 Career Services Policy.pdf`
  - Top 1 Retrieved: `PlacementPloicy/OJT Internship policy 2019-2020 onwards.pdf`

Both colliding documents reside within the **exact same folder and domain (`PlacementPloicy`)**! Domain-level metadata filtering cannot disambiguate between two documents belonging to the same policy domain.

---

## 4. Why Automated Filtering Degrades Performance (The False-Negative Trap)

University policies have inherently overlapping boundaries:
- **Internship vs Placement PPO:** A student asking about pre-placement offers during internships touches both `ojt` and `placement_career`.
- **Academic Benefits vs Career Services:** Academic benefits granted for competitive exams or industry certifications are documented inside the Career Services folder.
- **Dress Code vs Placement:** Dress code guidelines for placement drives appear in both the Dress Code policy and Career Services policy.

### Failure Case Study: Q035
- **Query:** *"Under the Internship-with-PPO academic benefit, what is the stipend range...?"*
- **Ground Truth Source:** `PlacementPloicy/Academic Benefit Plan - Placement and Internship...pdf` (`placement_career`)
- **Router Prediction:** `ojt` (triggered by the word *"Internship"*)
- **Impact of Hard Filter:**
  - **Unfiltered:** Rank 1 hit (`1.000`)
  - **Filtered:** **Zero recall (`0.000`)**! The correct document was strictly excluded from the search pool by the metadata filter.

---

## 5. Architectural Decision & Guidelines

1. **No Automated Hard Filtering:**
   The backend retrieval pipeline (`app/query/retriever.py` and `evaluation/lib/harness.py`) must continue to execute unconstrained dense vector search across the full corpus.
2. **Soft Reranking is Superior:**
   The Cohere cross-encoder (`rerank-v3.5`) already performs cross-domain semantic scoring without the risk of false-negative exclusions.
3. **Role of Metadata in UniAssist:**
   - Document domain metadata (`library`, `hostel`, `placement`, etc.) should be maintained for **UI display badges** and **citation transparency**.
   - If user-facing facet selection is implemented in the frontend (e.g. user explicitly toggles a *"Search only Hostels"* tab), filtering may be applied on user request only.
