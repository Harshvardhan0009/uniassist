# LLM Matched-Subset Analysis — Gemini 3.6 Flash vs Groq GPT-OSS-120B

> **Purpose:** Recompute LLM evaluation metrics and hallucination audits on the **identical intersection of question IDs**
> to eliminate sample-size asymmetries (Gemini $n=18$ vs Groq $n=63$) present in earlier phase reports.

---

## 1. Sample Structure & Question Intersection

| Dimension | Gemini Run (`rag_e5_1500_gemini`) | Groq Run (`rag_e5_1500_groq_gptoss`) | Matched Subset Intersection |
|---|---:|---:|---:|
| **Total Scored Questions** | 18 | 63 | **18** |
| **Answerable Questions** | 18 | 57 | **18** |
| **Unanswerable Questions** | 0 | 6 | **0** |
| **Question IDs** | Q001–Q018 | Q001–Q063 | **Q001–Q018** |

> [!NOTE]
> The primary scored Gemini run (`rag_e5_1500_gemini_full_rag_scored.json`) evaluated exactly 18 questions before exhausting Google's free-tier daily cap (20 requests/day). All 18 are answerable questions. The 6 unanswerables (`Q056`–`Q061`) were evaluated separately in Phase 18 (`abstention_experiment_raw.json`).

---

## 2. Head-to-Head Metrics on Identical Matched Subset ($n=18$)

Both models evaluated against the identical frozen retrieval context (E5-base-v2, 1500/200, Cohere top-5):

| Metric | Gemini 3.6 Flash ($n=18$) | Groq GPT-OSS-120B ($n=18$) | Relative Delta | Matched Winner |
|---|---:|---:|---:|---|
| **Token Recall vs Ground Truth** | **0.7679** | 0.5917 | **+29.78%** | **Gemini** |
| **Token F1 vs Ground Truth** | **0.5739** | 0.4877 | **+17.67%** | **Gemini** |
| **Numerical Fact Recall** | **0.9259** | 0.8518 | **+8.7%** | **Gemini** |
| **Citation F1** | 0.8241 | 0.8241 | +0.0000 | **Exact Tie** |
| **Citation Precision** | 0.7500 | 0.7500 | 0.0000 | **Tie** |
| **Citation Recall** | 1.0000 | 1.0000 | 0.0000 | **Tie** |
| **Avg Generation Latency** | **3902.0 ms** | 11950.8 ms | **3.06× faster** | **Gemini** |
| **False Abstentions on Answerables** | **0 / 18** | **0 / 18** | 0.0% | **Tie** |

---

## 3. Faithfulness & Hallucination Audit on Matched Subset ($n=18$)

Audit records from [`faithfulness_audit_scored.json`](../experiments/results/faithfulness_audit_scored.json) filtered to the identical 18 questions (Q001–Q018):

| Metric | Baseline V1 (MiniLM, uncalibrated) | Groq GPT-OSS-120B (E5, calibrated) | Gemini 3.6 Flash (E5, calibrated) |
|---|---:|---:|---:|
| **Sample Size** | 18 Qs (Q001–Q018) | 18 Qs (Q001–Q018) | 18 Qs (Q001–Q018) |
| **Strictly Faithful Answers** | 6 / 18 (33.3%) | 12 / 18 (66.7%) | **16 / 18 (88.9%)** |
| **Numerical Hallucinations** | 8 errors | **2 errors** (`Q002`, `Q011`) | **2 errors** (`Q004`, `Q011`) |
| **Mean Sentence Groundedness** | 0.5463 | 0.5386 | **0.6914** |

> [!IMPORTANT]
> **Resolution of the 28 → 9 → 2 Hallucination Discrepancy:**
> - The earlier Phase 20 report stated: *28 errors in Baseline $\to$ 9 in Groq $\to$ 2 in Gemini*.
> - That comparison was **unmatched** (Baseline $n=63$, Groq $n=63$, Gemini $n=18$).
> - On the **identical 18-question matched subset**, Baseline has **8 errors**, Groq has **2 errors**, and Gemini has **2 errors**.
> - Groq and Gemini are **exactly tied** on numerical hallucination count (2 errors each) when measured on the same questions.

---

## 4. Safety & Out-of-Domain Abstention ($n=6$)

Because Q001–Q018 contain only answerable questions, abstention was verified using Phase 18's adversarial unanswerable test set (`Q056`–`Q061`):

| Model | True Refusals ($n=6$) | Hallucinations | Abstention Rate |
|---|---:|---:|---:|
| **Gemini 3.6 Flash** | **6 / 6** | 0 / 6 | **100.0%** |
| **Groq GPT-OSS-120B** | **6 / 6** | 0 / 6 | **100.0%** |

---

## 5. Statistical Adequacy & Scientific Conclusion

1. **Does the Phase 12 Conclusion Hold?**
   **Yes.** On the matched 18 questions, Gemini 3.6 Flash demonstrates:
   - +29.77% higher token recall against reference answers
   - +8.69% higher numerical fact recall
   - 3.06× faster generation latency (3.90s vs 11.95s)
   - Exact tie on citation F1 (0.8241) and numerical hallucinations (2 errors each).

2. **Is $n=18$ Large Enough to Support It Statistically?**
   - **$n=18$ is a small sample.** It covers only 31.6% of the 57 answerable benchmark questions.
   - While the quality advantage (+29.8% token recall) and latency gap (3.1×) are wide enough to be practically meaningful, the sample size has broad confidence intervals and did not include conversational follow-ups (tested later in Phase 15).
   - Therefore, the architectural decision made in Phase 12—**retaining Gemini 3.6 Flash as primary for speed and phrasing, with Groq GPT-OSS-120B as automatic zero-downtime fallback**—remains the most robust operational configuration.
