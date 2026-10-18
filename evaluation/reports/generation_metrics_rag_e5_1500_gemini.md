# Generation Metrics — rag_e5_1500_gemini_full_rag

- **Model:** `gemini-3.6-flash` (google-gemini)
- **Evaluated:** 18 questions (18 answerable, 0 unanswerable)

## 1. Summary Metrics

| Metric Area | Metric | Score |
|---|---|---|
| **Citation** | Precision | **0.750** |
| | Recall | **1.000** |
| | F1 Score | **0.824** |
| **Answer Correctness** | Token Recall vs GT | **0.768** |
| | Token F1 vs GT | **0.574** |
| | Numerical Fact Recall | **0.926** |
| | False Abstentions | **0/18** |
| **Safety & Abstention** | Abstention Rate (Unanswerable) | **100.0%** (0/0) |
| | Hallucination Rate (Unanswerable) | **0.0%** (0/0) |

## 2. Latency (ms)

| Step | Avg (ms) | P50 (ms) | P95 (ms) | Max (ms) |
|---|---|---|---|---|
| Generation | 3901.99 | 3435.66 | 6418.91 | 7755.52 |
| End-to-End Total | 4543.67 | 3980.69 | 7127.15 | 8465.28 |

## 3. By Category (Answerable Questions)

| Category | n | Citation F1 | Token Recall | Token F1 | Num Fact Recall |
|---|---|---|---|---|---|
| `academic` | 2 | 0.750 | 0.624 | 0.548 | 0.833 |
| `direct_factual` | 8 | 0.812 | 0.783 | 0.582 | 0.875 |
| `policy` | 8 | 0.854 | 0.789 | 0.573 | 1.000 |
