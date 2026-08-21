"""Metrics package.

Phase 4 builds the runners that emit *raw* retrieval/answer records.
Phase 5 implements `retrieval_metrics.py` (Recall@1/5/10/20, MRR, Precision@K,
Hit Rate@K) and Phase 13 implements `generation_metrics.py`. These consume the
raw artifacts produced by the runners so scoring is decoupled from execution.
"""
