"""
Query Chain — Full query orchestration (Steps 7–9).

Orchestrates: embed query → retrieve → rerank → generate answer.
Exposed as a single function for the API layer.
"""

import json
import logging
import time

from app.config import settings
from app.query.compressor import prune_candidate_chunks
from app.query.generator import generate_answer, stream_answer_tokens
from app.query.reranker import rerank
from app.query.retriever import retrieve
from app.query.rewriter import rewrite_query

logger = logging.getLogger(__name__)


def query(question: str, history: list[dict] | None = None) -> dict:
    """
    Run the full RAG query pipeline.

    Orchestration:
      1. Query Rewriter: Turn 2+ history resolution
      2. Retriever: Prefix-aware E5 dense search (Top-20)
      3. Reranker: Cohere cross-encoder reranking (Top-5)
      4. Compressor: Score-aware adaptive tail pruning
      5. Generator: Calibrated grounded generation with LLM failover

    Args:
        question: The user's natural language question.
        history: Optional prior conversation turns ([{role, content}, ...])
            used to give the generator follow-up context.

    Returns:
        Dict with answer, sources, timing breakdown, and candidate metrics.
    """
    timings = {}
    start = time.time()

    # ── Step 1: Query Rewriting (Turn 2+ only) ────────────────────────
    search_query = question
    if history and settings.ENABLE_QUERY_REWRITING:
        search_query, rewrite_ms = rewrite_query(question, history)
        timings["rewriter"] = round(rewrite_ms / 1000.0, 3)

    # ── Step 2: Retrieve ─────────────────────────────────────────────
    t0 = time.time()
    candidates = retrieve(search_query)
    timings["retrieval"] = round(time.time() - t0, 3)

    if not candidates:
        return {
            "answer": "I do not have sufficient information in the official university documents to answer this question.",
            "sources": [],
            "has_llm": False,
            "timing": timings,
            "candidates_retrieved": 0,
            "candidates_reranked": 0,
            "candidates_compressed": 0,
        }

    # ── Step 3: Rerank ───────────────────────────────────────────────
    t0 = time.time()
    reranked = rerank(search_query, candidates)
    timings["reranking"] = round(time.time() - t0, 3)

    # ── Step 4: Context Compression & Trimming ───────────────────────
    t0 = time.time()
    if settings.ENABLE_CONTEXT_COMPRESSION:
        final_context_docs = prune_candidate_chunks(reranked)
    else:
        final_context_docs = reranked
    timings["compression"] = round(time.time() - t0, 4)

    # ── Step 5: Generate ─────────────────────────────────────────────
    t0 = time.time()
    result = generate_answer(question, final_context_docs, history=history)
    timings["generation"] = round(time.time() - t0, 3)

    timings["total"] = round(time.time() - start, 3)

    return {
        **result,
        "search_query": search_query if search_query != question else None,
        "timing": timings,
        "candidates_retrieved": len(candidates),
        "candidates_reranked": len(reranked),
        "candidates_compressed": len(final_context_docs),
    }


def stream_query(question: str, history: list[dict] | None = None):
    """
    Run the full RAG query pipeline and yield Server-Sent Events (SSE) in real time.

    Events:
      - event: metadata -> JSON with sources, search_query, chunk counts
      - event: token -> JSON with each incremental token as it arrives
      - event: done -> JSON with full timings breakdown
    """
    timings = {}
    start = time.time()

    # Step 1: Query Rewriting
    search_query = question
    if history and settings.ENABLE_QUERY_REWRITING:
        search_query, rewrite_ms = rewrite_query(question, history)
        timings["rewriter"] = round(rewrite_ms / 1000.0, 3)

    # Step 2: Retrieve
    t0 = time.time()
    candidates = retrieve(search_query)
    timings["retrieval"] = round(time.time() - t0, 3)

    if not candidates:
        yield f"event: metadata\ndata: {json.dumps({'sources': [], 'search_query': None, 'candidates_compressed': 0})}\n\n"
        yield f"event: token\ndata: {json.dumps({'token': 'I do not have sufficient information in the official university documents to answer this question.'})}\n\n"
        timings["total"] = round(time.time() - start, 3)
        yield f"event: done\ndata: {json.dumps({'timing': timings})}\n\n"
        return

    # Step 3: Rerank
    t0 = time.time()
    reranked = rerank(search_query, candidates)
    timings["reranking"] = round(time.time() - t0, 3)

    # Step 4: Context Compression
    t0 = time.time()
    if settings.ENABLE_CONTEXT_COMPRESSION:
        final_context_docs = prune_candidate_chunks(reranked)
    else:
        final_context_docs = reranked
    timings["compression"] = round(time.time() - t0, 4)

    sources = list(dict.fromkeys(doc.metadata.get("source_file", "Unknown") for doc in final_context_docs))

    # Send metadata event first
    metadata_payload = {
        "sources": sources,
        "search_query": search_query if search_query != question else None,
        "candidates_retrieved": len(candidates),
        "candidates_reranked": len(reranked),
        "candidates_compressed": len(final_context_docs),
    }
    yield f"event: metadata\ndata: {json.dumps(metadata_payload)}\n\n"

    # Step 5: Stream answer tokens
    t0 = time.time()
    for token in stream_answer_tokens(question, final_context_docs, history=history):
        yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
    timings["generation"] = round(time.time() - t0, 3)
    timings["total"] = round(time.time() - start, 3)

    yield f"event: done\ndata: {json.dumps({'timing': timings})}\n\n"
