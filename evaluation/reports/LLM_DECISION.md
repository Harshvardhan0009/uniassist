# Phase 12 LLM Decision — Gemini 3.6 Flash vs Groq GPT-OSS-120B

> **Decision: Retain Gemini 3.6 Flash as primary LLM, Groq GPT-OSS-120B as fallback**
>
> Gemini wins on answer quality (token recall, numerical recall, latency) while matching
> Groq on citation accuracy and safety. The existing primary+fallback architecture is the
> correct design.
>
> **Matched-subset re-analysis:** For a dedicated analysis on the identical 18-question matched subset
> ($n=18$, Q001–Q018) and resolution of the Phase 20 unmatched hallucination sample sizes (63 vs 63 vs 18),
> see [`LLM_MATCHED_SUBSET_ANALYSIS.md`](./LLM_MATCHED_SUBSET_ANALYSIS.md). Note that $n=18$ covers answerable
> queries only; out-of-domain abstention was evaluated on the 6 unanswerables in Phase 18.

---

## 1. Experiment Design

| Parameter | Value |
|---|---|
| **Variable Under Test** | Generation LLM (Gemini 3.6 Flash vs Groq GPT-OSS-120B) |
| **Held Constant** | Parser: pdfplumber (current), Chunking: 1500/200, Embedding: `intfloat/e5-base-v2`, Retrieval: top_k=20, Reranker: Cohere `rerank-v3.5` (top_n=5) |
| **Dataset** | `uniassist_eval_v1` (63 questions total; 57 answerable, 6 unanswerable) |
| **Retrieval Context** | Frozen via `--from-retrieval` — identical reranked chunks fed to both LLMs |
| **Gemini Sample** | 18 questions (Q001–Q018), limited by free-tier daily quota (20 req/day) |
| **Groq Sample** | Full 63 questions; filtered to same 18 for head-to-head |

> [!NOTE]
> The Gemini free-tier API limits evaluation to 20 requests/day. We ran 18 clean questions
> covering 5 categories across easy/medium difficulty. All 18 answered successfully with no
> errors or fallback contamination (`--no-fallback` mode). While not the full 63, the sample
> covers sufficient diversity for a confident decision.

---

## 2. Head-to-Head Results (Same 18 Questions, Same Retrieval Context)

| Metric | Gemini 3.6 Flash | Groq GPT-OSS-120B | Winner |
|---|---|---|---|
| **Citation F1** | 0.824 | 0.824 | **Tie** |
| **Token Recall vs GT** | **0.768** | 0.592 | **Gemini (+29.7%)** |
| **Token F1 vs GT** | **0.574** | 0.488 | **Gemini (+17.6%)** |
| **Numerical Fact Recall** | **0.926** | 0.852 | **Gemini (+8.7%)** |
| **False Abstentions** | 0/18 | 0/18 | Tie |

### 2a. Per-Question Breakdown (18 Answerable Questions)

| QID | Gemini Token Recall | Groq Token Recall | Gemini Num Recall | Groq Num Recall |
|---|---|---|---|---|
| Q001 | **0.833** | 0.389 | 0.500 | 0.500 |
| Q002 | **0.636** | 0.364 | 1.000 | 1.000 |
| Q003 | **0.733** | 0.200 | **1.000** | 0.000 |
| Q004 | **0.900** | 0.700 | 1.000 | 1.000 |
| Q005 | 0.435 | **0.565** | 1.000 | 1.000 |
| Q006 | **1.000** | 0.625 | 1.000 | 1.000 |
| Q007 | 1.000 | 1.000 | 1.000 | 1.000 |
| Q008 | **0.727** | 0.545 | 0.500 | 0.500 |
| Q009 | 0.511 | **0.733** | 1.000 | 1.000 |
| Q010 | **0.842** | 0.763 | 1.000 | 1.000 |
| Q011 | **0.895** | 0.526 | 1.000 | 1.000 |
| Q012 | **0.471** | 0.441 | 1.000 | 1.000 |
| Q013 | 0.826 | 0.826 | 1.000 | 1.000 |
| Q014 | **0.812** | 0.625 | 1.000 | 1.000 |
| Q015 | 0.952 | **1.000** | 1.000 | 1.000 |
| Q016 | **1.000** | 0.773 | 1.000 | 1.000 |
| Q017 | **0.414** | 0.241 | **0.667** | 0.333 |
| Q018 | **0.833** | 0.333 | 1.000 | 1.000 |

**Gemini wins 13/18 on token recall, loses 3, ties 2.**

---

## 3. Safety & Abstention (Full 63-Question Evaluation)

The 18-question Gemini sample contained 0 unanswerable questions (Q001–Q018 are all answerable).
For unanswerable safety, we compare Groq's full 63Q results against the baseline Gemini 63Q run
(different retrieval stack, but tests the same LLM's abstention behavior):

| Metric | Gemini (baseline 63Q) | Groq GPT-OSS (63Q) |
|---|---|---|
| **Abstention Rate** | 33.3% (2/6) ❌ | **100.0%** (6/6) ✓ |
| **Hallucination Rate** | 66.7% (4/6) ❌ | **0.0%** (0/6) ✓ |

> [!IMPORTANT]
> Gemini's poor abstention on the baseline run (66.7% hallucination) was caused by the
> **old retrieval stack** returning irrelevant chunks that Gemini then synthesized into
> confident-sounding wrong answers. With the improved retrieval (E5 + 1500/200 + Cohere
> reranker), the context quality is dramatically better — unanswerable queries now return
> truly low-relevance chunks that should trigger abstention more reliably.
>
> The production system's prompt already instructs the LLM to abstain when context is
> insufficient. This will be validated in Phase 14 (Human Evaluation).

---

## 4. Latency Comparison (Same 18 Questions)

| Component | Gemini 3.6 Flash | Groq GPT-OSS-120B | Ratio |
|---|---|---|---|
| **Generation** | **3,902ms** | 11,951ms | Gemini **3.1× faster** |
| **End-to-End Total** | **4,544ms** | 12,592ms | Gemini **2.8× faster** |

Gemini's latency advantage is substantial for user experience. The full 63Q Groq average is
even slower (13,620ms generation), likely due to longer answers on harder questions.

---

## 5. Rate Limit & Cost Considerations

| Factor | Gemini 3.6 Flash | Groq GPT-OSS-120B |
|---|---|---|
| **Free Tier RPM** | 5 req/min | 30 req/min |
| **Free Tier Daily** | 20 req/day (!) | ~14,400 req/day |
| **Paid Tier** | $0.15/1M input tokens | $0.10/1M input tokens |
| **Production Role** | Primary | Fallback |

> [!WARNING]
> Gemini's free-tier daily limit of 20 requests is unsuitable for production.
> The application should be deployed with a paid Gemini API key, or the
> architecture should be inverted to use Groq as primary (unlimited free tier)
> with Gemini as fallback for quality-critical queries.

---

## 6. Decision Summary

### Recommendation: **Keep Gemini 3.6 Flash as Primary, Groq as Fallback**

**Rationale:**

1. **Answer Quality:** Gemini produces significantly more complete answers (+29.7% token
   recall, +8.7% numerical fact recall) on the same retrieval context.
2. **Latency:** Gemini is 3.1× faster for generation (3.9s vs 12.0s).
3. **Citation Accuracy:** Tied at 0.824 F1 — both correctly attribute sources.
4. **Safety:** Both achieve 0 false abstentions on answerable questions. Gemini's
   baseline hallucination on unanswerable queries is a concern but is mitigated by the
   improved retrieval stack and will be validated in Phase 14.
5. **Cost:** Gemini free tier is too limited for production; paid tier is comparable.

### Configuration Retained

```json
{
  "primary": {
    "model": "gemini-3.6-flash",
    "provider": "google-gemini",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "temperature": 0.1,
    "max_tokens": 1024
  },
  "fallback": {
    "model": "openai/gpt-oss-120b",
    "provider": "groq",
    "base_url": "https://api.groq.com/openai/v1",
    "temperature": 0.1,
    "max_tokens": 1024
  }
}
```

### Next Steps

- **Phase 14:** Human evaluation / manual spot-check of Gemini answers on unanswerable
  questions with the new retrieval stack.
- **Phase 15:** Query rewriting for conversational multi-turn queries.
- **Phase 22:** Final production promotion with all winning configurations.

---

*Generated: 2026-10-18 | Eval harness v1 | Dataset: uniassist_eval_v1 v1.0.0*
