# Generation Metrics — rag_e5_1500_groq_gptoss_full_rag

- **Model:** `openai/gpt-oss-120b` (groq)
- **Evaluated:** 63 questions (57 answerable, 6 unanswerable)

## 1. Summary Metrics

| Metric Area | Metric | Score |
|---|---|---|
| **Citation** | Precision | **0.719** |
| | Recall | **1.000** |
| | F1 Score | **0.807** |
| **Answer Correctness** | Token Recall vs GT | **0.619** |
| | Token F1 vs GT | **0.505** |
| | Numerical Fact Recall | **0.860** |
| | False Abstentions | **0/57** |
| **Safety & Abstention** | Abstention Rate (Unanswerable) | **100.0%** (6/6) |
| | Hallucination Rate (Unanswerable) | **0.0%** (0/6) |

## 2. Latency (ms)

| Step | Avg (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---|---|---|---|
| Generation | 13619.61 | 13826.85 | 20678.78 | 22995.26 |
| End-to-End Total | 14288.64 | 14449.87 | 21255.33 | 23491.38 |

## 3. By Category (Answerable Questions)

| Category | n | Citation F1 | Token Recall | Token F1 | Num Fact Recall |
|---|---|---|---|---|---|
| `academic` | 7 | 0.762 | 0.539 | 0.473 | 0.905 |
| `conversational` | 6 | 0.639 | 0.703 | 0.501 | 0.833 |
| `direct_factual` | 8 | 0.812 | 0.548 | 0.447 | 0.750 |
| `exact_terminology` | 7 | 0.762 | 0.830 | 0.619 | 1.000 |
| `multi_condition` | 6 | 0.889 | 0.539 | 0.468 | 0.708 |
| `placement` | 7 | 0.857 | 0.562 | 0.441 | 0.905 |
| `policy` | 9 | 0.870 | 0.724 | 0.584 | 1.000 |
| `table_based` | 7 | 0.833 | 0.488 | 0.489 | 0.729 |
