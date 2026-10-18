# Generation Metrics — baseline_v1_gemini36_full_rag

- **Model:** `gemini-3.6-flash` (default)
- **Evaluated:** 63 questions (57 answerable, 6 unanswerable)

## 1. Summary Metrics

| Metric Area | Metric | Score |
|---|---|---|
| **Citation** | Precision | **0.637** |
| | Recall | **0.983** |
| | F1 Score | **0.746** |
| **Answer Correctness** | Token Recall vs GT | **0.756** |
| | Token F1 vs GT | **0.241** |
| | Numerical Fact Recall | **0.774** |
| | False Abstentions | **0/57** |
| **Safety & Abstention** | Abstention Rate (Unanswerable) | **33.3%** (2/6) |
| | Hallucination Rate (Unanswerable) | **66.7%** (4/6) |

## 2. Latency (ms)

| Step | Avg (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---|---|---|---|
| Generation | 2726.5 | 2024.11 | 6387.42 | 8079.65 |
| End-to-End Total | 3398.62 | 2677.03 | 7251.14 | 8635.97 |

## 3. By Category (Answerable Questions)

| Category | n | Citation F1 | Token Recall | Token F1 | Num Fact Recall |
|---|---|---|---|---|---|
| `academic` | 7 | 0.833 | 0.823 | 0.249 | 1.000 |
| `conversational` | 6 | 0.583 | 0.797 | 0.098 | 0.667 |
| `direct_factual` | 8 | 0.750 | 0.813 | 0.594 | 0.750 |
| `exact_terminology` | 7 | 0.738 | 0.782 | 0.056 | 1.000 |
| `multi_condition` | 6 | 0.806 | 0.569 | 0.149 | 0.555 |
| `placement` | 7 | 0.714 | 0.755 | 0.143 | 0.714 |
| `policy` | 9 | 0.833 | 0.851 | 0.425 | 1.000 |
| `table_based` | 7 | 0.667 | 0.600 | 0.076 | 0.395 |
