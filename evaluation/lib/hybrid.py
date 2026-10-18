"""Hybrid Search module (Phase 17) — Dense E5 + Sparse BM25 via Reciprocal Rank Fusion (RRF).

Combines dense semantic vector retrieval (Chroma + intfloat/e5-base-v2) with
lexical keyword search (rank-bm25) over the exact frozen corpus snapshot.
Fuses candidate rankings using Reciprocal Rank Fusion (RRF, default k=60).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from . import paths, snapshot as snap
from .harness import chunk_id, retrieve_candidates

logger = logging.getLogger(__name__)

# Cache of initialized BM25 retrievers keyed by snapshot name
_BM25_CACHE: dict[str, BM25Retriever] = {}


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """In-memory BM25 lexical retriever over a snapshot corpus."""

    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.tokenized_corpus = [tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 20) -> tuple[list[Document], float]:
        """Perform BM25 search. Returns (candidates, latency_ms)."""
        t0 = time.perf_counter()
        tokens = tokenize(query)
        if not tokens:
            return [], 0.0

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        candidates: list[Document] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            doc = self.docs[idx]
            # Copy document to avoid clobbering metadata across queries
            copied = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            copied.metadata["bm25_score"] = round(score, 4)
            candidates.append(copied)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return candidates, latency_ms


def get_bm25_retriever(snapshot_name: str) -> BM25Retriever:
    """Retrieve or build a cached BM25Retriever for the given snapshot."""
    global _BM25_CACHE
    if snapshot_name not in _BM25_CACHE:
        logger.info("Initializing BM25 index for snapshot '%s'...", snapshot_name)
        docs = snap.load_snapshot(snapshot_name)
        _BM25_CACHE[snapshot_name] = BM25Retriever(docs)
    return _BM25_CACHE[snapshot_name]


def reciprocal_rank_fusion(
    dense_candidates: list[Document],
    sparse_candidates: list[Document],
    top_k: int = 20,
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[Document]:
    """Fuse dense and sparse candidate lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = dense_weight / (rrf_k + rank_dense) + sparse_weight / (rrf_k + rank_sparse)
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_candidates, start=1):
        cid = chunk_id(doc)
        scores[cid] = scores.get(cid, 0.0) + (dense_weight / (rrf_k + rank))
        if cid not in doc_map:
            doc_map[cid] = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
        else:
            doc_map[cid].metadata["retrieval_cosine_score"] = doc.metadata.get("retrieval_cosine_score")

    for rank, doc in enumerate(sparse_candidates, start=1):
        cid = chunk_id(doc)
        scores[cid] = scores.get(cid, 0.0) + (sparse_weight / (rrf_k + rank))
        if cid not in doc_map:
            doc_map[cid] = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
        doc_map[cid].metadata["bm25_score"] = doc.metadata.get("bm25_score")

    sorted_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[:top_k]

    fused_docs: list[Document] = []
    for cid in sorted_cids:
        d = doc_map[cid]
        d.metadata["rrf_score"] = round(scores[cid], 6)
        fused_docs.append(d)

    return fused_docs


def retrieve_hybrid_candidates(
    store: Chroma,
    snapshot_name: str,
    query: str,
    top_k: int = 20,
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    min_score: float = 0.0,
) -> tuple[list[Document], float, dict[str, float]]:
    """Execute hybrid retrieval (Dense E5 + BM25 + RRF).

    Returns (fused_docs, total_latency_ms, latency_breakdown).
    """
    t0 = time.perf_counter()

    # 1. Dense retrieval
    dense_docs, dense_lat = retrieve_candidates(store, query, top_k=top_k, min_score=min_score)

    # 2. Sparse BM25 retrieval
    bm25 = get_bm25_retriever(snapshot_name)
    sparse_docs, sparse_lat = bm25.search(query, top_k=top_k)

    # 3. Reciprocal Rank Fusion
    t_rrf = time.perf_counter()
    fused_docs = reciprocal_rank_fusion(
        dense_candidates=dense_docs,
        sparse_candidates=sparse_docs,
        top_k=top_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    rrf_lat = round((time.perf_counter() - t_rrf) * 1000, 2)
    total_lat = round((time.perf_counter() - t0) * 1000, 2)

    latency_breakdown = {
        "dense_ms": dense_lat,
        "sparse_ms": sparse_lat,
        "rrf_ms": rrf_lat,
        "total_ms": total_lat,
    }
    return fused_docs, total_lat, latency_breakdown
