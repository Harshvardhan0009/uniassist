# Chunking decision (Phase 10)

**Date:** 2026-08-30 · **Status:** ✅ **SELECTED — 1500 / 200** (chunk size / overlap). Promotion to production is deferred to **Phase 22** (with the rest of the winning config). · **Dataset:** `uniassist_eval_v1` (57 answerable) · **Method:** dense retrieval, `intfloat/e5-base-v2` held constant, `top_k=20`, cosine, no rerank; only chunk size/overlap changes (fresh frozen snapshot per size, raw text).

Full tables: [`CHUNKING_COMPARISON.md`](./CHUNKING_COMPARISON.md). Raw: `experiments/results/chunking_comparison.json`.

## Result

| chunk / overlap | #chunks | src R@1 | src MRR | **page R@1** | page MRR |
|---|--:|--:|--:|--:|--:|
| 1000 / 150 | 384 | 0.912 | 0.939 | 0.702 | 0.776 |
| **1500 / 200** | 285 | **0.930** | **0.952** | **0.789** | **0.819** |
| 2500 / 300 (current) | 184 | 0.877 | 0.927 | 0.649 | 0.743 |
| 3500 / 400 | 159 | 0.860 | 0.917 | 0.649 | 0.741 |

## Findings

- **1500/200 is the best chunk size**, improving the current 2500/300 on source R@1 (0.877 → 0.930),
  source MRR (0.927 → 0.952), and most of all **page-level** precision (R@1 **0.649 → 0.789**, MRR
  0.743 → 0.819). Page-level matters directly for citation accuracy (Phase 20).
- **Larger is worse.** 2500/300 and 3500/400 are the weakest. E5-base has a **512-token limit
  (~2000 chars)**, so 2500–3500-char chunks are **truncated** — their tail text never gets embedded —
  and a larger chunk dilutes which page a hit points to. This is the mechanism behind the page-level drop.
- **By category (source level):** table / policy / multi-condition / direct-factual / academic are
  already perfect (1.000) at both 2500 and 1500, so 1500/200 **holds** them while lifting the weaker
  categories: exact_terminology 0.714 → 0.857, placement 0.714 → 0.857, conversational 0.500 → 0.667.
  No category regresses.
- **1000/150** is also better than baseline but slightly below 1500/200 (more, smaller chunks →
  marginally lower source R@1 and page R@1); 1500/200 is the sweet spot of completeness vs granularity.

## Recommendation

**Adopt 1500 / 200** as the chunk size/overlap for the E5-based system.

### Caveats / notes
- These are **dense** numbers. The reranker (constant) sits downstream; better dense candidates can
  only help it. A reranked + answer-quality re-confirmation can be run before Phase 22 promotion.
- The chunk-size effect is **embedding-dependent** via the token limit. The current production
  embedding (MiniLM) truncates even harder (256 tokens ≈ 1000 chars), so if E5 is *not* promoted,
  re-run this sweep for the deployed embedding. Since Phase 8 selected E5, 1500/200 is tuned for it.
- **Promotion (Phase 22):** change `chunker.py` `CHUNK_SIZE/CHUNK_OVERLAP` to 1500/200 (or drive from
  config) and **re-index** production. Not done now, per the plan's "don't modify production until the
  full winning config is chosen" rule.
