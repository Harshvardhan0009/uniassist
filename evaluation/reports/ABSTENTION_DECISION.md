# Phase 18 Decision Record: Abstention & Refusal Calibration ("I Don't Know" Robustness)

**Date:** 2026-10-18  
**Experiment:** Abstention Calibration across Primary (`gemini-3.6-flash`) and Fallback (`openai/gpt-oss-120b`) LLMs  
**Dataset:** `uniassist_eval_v1` (63 questions: 57 answerable, 6 unanswerable adversarial items `Q056`–`Q061`)  
**Retriever Stack:** `intfloat/e5-base-v2` (1500/200) + Cohere `rerank-v3.5` (top-20 → top-5)  
**Raw Results:** [`abstention_experiment_raw.json`](../experiments/results/abstention_experiment_raw.json)  
**Gating Sweep Data:** [`abstention_gating_sweep.json`](../experiments/results/abstention_gating_sweep.json)  

---

## 1. Executive Summary & Decision

| Metric | Prompt-Based (Gemini 3.6 Flash) | Prompt-Based (Groq GPT-OSS-120B) | Pre-LLM Score Gating ($\tau = 0.50$) | Pre-LLM Score Gating ($\tau = 0.80$) |
| :--- | :---: | :---: | :---: | :---: |
| **True Refusals (Unanswerable caught)** | **6 / 6 (100.0%)** | **6 / 6 (100.0%)** | 3 / 6 (50.0%) | **6 / 6 (100.0%)** |
| **False Refusals (Answerable denied)** | **0 / 57 (0.0%)** | **0 / 57 (0.0%)** | 5 / 57 (8.8%) | 19 / 57 (33.3%) |
| **Refusal Precision** | **100.0%** | **100.0%** | 37.5% | 24.0% |
| **Refusal Recall** | **100.0%** | **100.0%** | 50.0% | **100.0%** |
| **Refusal F1 Score** | **1.000** | **1.000** | 0.429 | 0.387 |
| **Answerable Question Pass Rate** | **100.0% (57/57)** | **100.0% (57/57)** | 91.2% (52/57) | 66.7% (38/57) |
| **Hallucination Count on Out-of-Corpus** | **0** | **0** | N/A | N/A |

### **Official Decision: ADOPT Prompt-Based LLM Grounded Abstention; REJECT Pre-LLM Confidence Score Gating**

1. **100% Prompt-Based Abstention Integrity**: Both `gemini-3.6-flash` and `openai/gpt-oss-120b` achieve **100% refusal accuracy (6/6)** on adversarial out-of-corpus queries without fabricating answers or hallucinating policy details.
2. **Failure of Pre-LLM Score Gating**:
   - Setting a reranker threshold high enough to catch all unanswerable questions ($\tau = 0.80$) causes an intolerable **33.3% False Refusal Rate**, mistakenly rejecting 19 out of 57 valid student questions before they ever reach the generator.
   - Setting a moderate threshold ($\tau = 0.50$) misses half of the unanswerable queries (50% refusal recall) while still erroneously rejecting 8.8% of answerable questions.
   - **Reason**: Topical overlap causes out-of-scope questions (e.g., Q061 on tuition fees) to score high rerank confidence (0.7831) against hostel/placement fee tables, while terse but answerable policy clauses score as low as 0.1848.
3. **Nuanced Contextual Grounding**: LLM prompt-based abstention excels because models can identify partial topical overlap and explicitly clarify boundaries (e.g. Gemini on Q061: *"The provided context does not contain the annual tuition fee for B.Tech CSE; it only lists the Professional Enhancement Program (PEP) fee of Rs. 12,000"*).
4. **Safety Floor**: Maintain a minimal retrieval score floor only at $\tau = 0.0$ to ensure unconstrained dense candidate generation, relying on prompt grounding for faithful refusal.

---

## 2. Evaluation Across Adversarial Unanswerable Questions

The benchmark includes 6 adversarial out-of-corpus queries spanning temporal shifts, private administrative credentials, out-of-policy curriculum details, and absent fee tables:

| ID | Question | Top Retrieved Source | Top Rerank Score | Gemini 3.6 Flash Response | Groq GPT-OSS-120B Response | Verdict |
|---|---|---|:---:|---|---|:---:|
| **Q056** | What is today's mess menu? | `ResidentialFacilities/Charges For Residential Facilities.pdf` | 0.5487 | "The provided documents do not contain information regarding today's mess menu." | "The provided university documents do not contain the current mess menu." | **Clean Abstention** |
| **Q057** | Who is the current Vice Chancellor of LPU? | `NSSPolicy/LPUNSSPOLICY.pdf` | 0.3093 | "Based on the provided documents, the specific name of the current Vice Chancellor of Lovely Professional University is not mentioned." | "The provided documents do not contain the name of the current Vice Chancellor of LPU." | **Clean Abstention** |
| **Q058** | What is the last date to apply for admission to LPU in 2027? | `SemesterExchange/Semester_Year_Abroad_Policy.pdf` | 0.5143 | "The provided context does not contain information regarding the last date to apply for admission to LPU in 2027." | "The provided documents do not contain any information about the last date to apply for admission to LPU in 2027." | **Clean Abstention** |
| **Q059** | How many marks are required to pass the Data Structures end-term exam? | `PlacementPloicy/Academic Benefit Plan - Placement and Internship...` | 0.3068 | "Based on the provided documents, the specific passing marks required for the Data Structures end-term exam are not mentioned." | "The required passing marks for the Data Structures end-term exam are not provided in the given documents." | **Clean Abstention** |
| **Q060** | What is the Wi-Fi password for the hostel? | `ResidentialFacilities/Charges For Residential Facilities.pdf` | 0.0849 | "The provided context does not contain information regarding the Wi-Fi password for the hostel." | "The Wi-Fi password for the hostel is not provided in the available university documents." | **Clean Abstention** |
| **Q061** | What is the annual tuition fee for the B.Tech CSE program? | `PlacementPloicy/2023 Career Services Policy.pdf` | 0.7831 | "The provided context does not contain information about the annual tuition fee for the B.Tech CSE program. It only mentions the Professional Enhancement Program (PEP) fee..." | "The provided documents do not contain information about the annual tuition fee for the B.Tech CSE program. (Only PEP fees are listed)." | **Clean Abstention** |

---

## 3. Threshold Gating Sweep (Why Pre-LLM Refusal Fails)

To determine whether pre-LLM threshold gating could reduce LLM costs by preemptively rejecting out-of-scope questions, we swept reranker score cutoffs $\tau \in [0.10, 0.85]$ across all 63 benchmark questions:

| Threshold ($\tau$) | True Refusals (Unanswerable Caught) | False Refusals (Answerable Rejected) | Refusal Precision | Answerable Pass Rate | Impact on User Experience |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $\tau \ge 0.10$ | 1 / 6 (16.7%) | 0 / 57 (0.0%) | 100.0% | 100.0% | No false refusals, but catches only 1/6 unanswerables |
| $\tau \ge 0.20$ | 1 / 6 (16.7%) | 2 / 57 (3.5%) | 33.3% | 96.5% | Falsely refuses valid questions on terse annexures |
| $\tau \ge 0.40$ | 3 / 6 (50.0%) | 3 / 57 (5.3%) | 50.0% | 94.7% | Rejects 3 valid student queries |
| $\tau \ge 0.50$ | 3 / 6 (50.0%) | 5 / 57 (8.8%) | 37.5% | 91.2% | Rejects ~1 in 11 valid student queries |
| $\tau \ge 0.60$ | 5 / 6 (83.3%) | 6 / 57 (10.5%) | 45.5% | 89.5% | Rejects >10% of valid queries |
| $\tau \ge 0.70$ | 5 / 6 (83.3%) | 9 / 57 (15.8%) | 35.7% | 84.2% | Rejects ~1 in 6 valid queries |
| $\tau \ge 0.80$ | 6 / 6 (100.0%) | 19 / 57 (33.3%) | 24.0% | 66.7% | **Disastrous**: 1 in 3 legitimate questions rejected |
| $\tau \ge 0.85$ | 6 / 6 (100.0%) | 26 / 57 (45.6%) | 18.8% | 54.4% | Nearly half of valid queries rejected |

### Core Finding
Confidence score distributions between answerable and unanswerable questions heavily overlap:
- Lowest answerable top-rerank score: **0.1848**
- Highest unanswerable top-rerank score: **0.7831**

Because scalar similarity metrics cannot distinguish between "relevant terms in an irrelevant document" (Q061 fee table) and "exact answer in a low-density administrative clause", scalar threshold gating is fundamentally unviable.

---

## 4. Prompt Engineering Directives for Production

The production system prompt in `backend/app/query/generator.py` successfully enforces strict abstention via instruction 4:
> *"If the answer is genuinely NOT in the context, state that clearly."*

To maintain this high standard during Phase 22 promotion:
1. **Explicit Negative Constraint**: Retain the requirement that models never extrapolate or guess unstated numbers, dates, or credentials.
2. **Boundary Disclaimers**: Encourage models to acknowledge close topical matches (e.g. stating that PEP fees are listed when tuition fees are requested) rather than emitting a generic flat refusal, enhancing user trust.
3. **Extractive Fallback Integrity**: Ensure that if the LLM fails completely (e.g. rate limit), the extractive fallback clearly labels snippets under an explicit disclaimer banner rather than presenting raw text as an authoritative answer.
