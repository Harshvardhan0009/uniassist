"""
Query Chain — Full query orchestration (Steps 7–9).

Orchestrates: embed query → retrieve → rerank → generate answer.
Exposed as a single function for the API layer.
"""

import logging
import time

from app.query.generator import generate_answer
from app.query.reranker import rerank
from app.query.retriever import retrieve

logger = logging.getLogger(__name__)


def query(question: str, history: list[dict] | None = None) -> dict:
    """
    Run the full RAG query pipeline.

    Args:
        question: The user's natural language question.
        history: Optional prior conversation turns ([{role, content}, ...])
            used to give the generator follow-up context.

    Returns:
        Dict with:
          - answer: generated answer text
          - sources: list of source filenames
          - has_llm: whether LLM was used for generation
          - timing: dict with step timings in seconds
          - candidates_retrieved: number of initial candidates
          - candidates_reranked: number after reranking
    """
    timings = {}
    start = time.time()

    # ── Step 7: Retrieve ─────────────────────────────────────────────
    t0 = time.time()
    candidates = retrieve(question)
    timings["retrieval"] = round(time.time() - t0, 3)

    if not candidates:
        return {
            "answer": "I couldn't find any relevant information in the university documents for your question.",
            "sources": [],
            "has_llm": False,
            "timing": timings,
            "candidates_retrieved": 0,
            "candidates_reranked": 0,
        }

    # ── Step 8: Rerank ───────────────────────────────────────────────
    t0 = time.time()
    reranked = rerank(question, candidates)
    timings["reranking"] = round(time.time() - t0, 3)

    # ── Step 9: Generate ─────────────────────────────────────────────
    t0 = time.time()
    result = generate_answer(question, reranked, history=history)
    timings["generation"] = round(time.time() - t0, 3)

    timings["total"] = round(time.time() - start, 3)

    return {
        **result,
        "timing": timings,
        "candidates_retrieved": len(candidates),
        "candidates_reranked": len(reranked),
    }
