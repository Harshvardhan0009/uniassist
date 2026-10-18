# Phase 10 — Chunking experiments

Embedding **`intfloat/e5-base-v2`** (the Phase 8 selection) held constant; same corpus, `top_k=20`, cosine, dense-only (no rerank), 57 answerable. **Only chunk size/overlap changes.** Each size is a fresh frozen snapshot (raw text).

## Overall (source + page level)

| chunk size / overlap | #chunks | src R@1 | src R@5 | src R@10 | src MRR | page R@1 | page R@5 | page MRR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1000 / 150 | 384 | 0.912 | 1.000 | 1.000 | 0.939 | 0.702 | 0.860 | 0.776 |
| 1500 / 200 **←best** | 285 | 0.930 | 1.000 | 1.000 | 0.952 | 0.789 | 0.860 | 0.819 |
| 2500 / 300 (current) | 184 | 0.877 | 1.000 | 1.000 | 0.927 | 0.649 | 0.842 | 0.743 |
| 3500 / 400 | 159 | 0.860 | 0.983 | 1.000 | 0.917 | 0.649 | 0.842 | 0.741 |

**1500/200 wins** on source R@1/MRR and — most importantly — **page-level** (R@1 0.789 vs 0.649 for the current 2500/300; page MRR 0.819 vs 0.743). Larger chunks (2500, 3500) are weakest: E5-base truncates at 512 tokens (~2000 chars), so 2500–3500-char chunks lose their tail, and a bigger chunk dilutes which page a hit points to. 1000/150 is good but 1500/200 balances completeness vs granularity best.

## By category — current 2500/300 vs best 1500/200 (source level)

| Category | n | 2500/300 R@1 | 1500/200 R@1 | 2500/300 MRR | 1500/200 MRR |
|---|--:|--:|--:|--:|--:|
| table_based | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| policy | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_condition | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| direct_factual | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| exact_terminology | 7 | 0.714 | 0.857 | 0.857 | 0.905 |
| academic | 7 | 1.000 | 1.000 | 1.000 | 1.000 |
| placement | 7 | 0.714 | 0.857 | 0.821 | 0.929 |
| conversational | 6 | 0.500 | 0.667 | 0.681 | 0.742 |

> Table/policy/multi-condition are the document types the plan calls out. 1500/200 helps or holds across categories; the page-level gain is the headline (better citations, Phase 20).
