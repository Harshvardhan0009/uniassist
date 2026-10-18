"""
Context Compression & Adaptive Tail Pruning (Phase 19).

Reduces LLM prompt context length by:
1. Adaptive score-aware candidate pruning (drops irrelevant low-scoring tail chunks).
2. Structural whitespace and redundant empty table line cleanup.
3. Clean, page-aware source header formatting.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


def clean_chunk_text(text: str) -> str:
    """Clean redundant whitespace, excessive newlines, and empty lines from chunk text."""
    if not text:
        return ""
    # Normalize Windows CRLF to LF
    text = text.replace("\r\n", "\n")
    # Collapse 3 or more consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    # Remove lines that are just empty table cells e.g. "|  |  |  |"
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\|(\s*\|\s*)+$", stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def prune_candidate_chunks(
    documents: list[Document],
    min_rel_ratio: float | None = None,
    min_abs_score: float | None = None,
    min_chunks: int | None = None,
    max_chunks: int | None = None,
) -> list[Document]:
    """Prune low-confidence tail chunks after cross-encoder reranking.

    Rules:
    1. Always keep up to `min_chunks` (default 2) if available.
    2. For ranks > min_chunks: keep only if score >= min_abs_score and score >= (min_rel_ratio * top_score).
    3. Never exceed `max_chunks`.
    """
    if not documents:
        return []

    min_rel_ratio = settings.COMPRESSION_REL_RATIO if min_rel_ratio is None else min_rel_ratio
    min_abs_score = settings.COMPRESSION_MIN_SCORE if min_abs_score is None else min_abs_score
    min_chunks = settings.COMPRESSION_MIN_CHUNKS if min_chunks is None else min_chunks
    max_chunks = settings.RERANK_TOP_N if max_chunks is None else max_chunks

    top_doc = documents[0]
    top_score = float(top_doc.metadata.get("rerank_score", 1.0) or 1.0)

    # If top chunk has a low score (< 0.25), keep up to min_chunks for LLM context
    if top_score < 0.25:
        return documents[:min_chunks]

    kept: list[Document] = []
    for idx, doc in enumerate(documents, start=1):
        if idx <= min_chunks:
            kept.append(doc)
            continue

        score = float(doc.metadata.get("rerank_score", 0.0) or 0.0)
        # Keep if meets absolute and relative relevance threshold
        if score >= min_abs_score and score >= (min_rel_ratio * top_score):
            kept.append(doc)

        if len(kept) >= max_chunks:
            break

    logger.debug(
        "Pruned %d reranked chunks → %d compressed chunks",
        len(documents),
        len(kept),
    )
    return kept


def format_compressed_context(
    documents: list[Document],
    enable_pruning: bool = True,
) -> tuple[str, list[Document]]:
    """Format documents into a compressed context prompt block with page-aware attribution."""
    if enable_pruning and settings.ENABLE_CONTEXT_COMPRESSION:
        pruned_docs = prune_candidate_chunks(documents)
    else:
        pruned_docs = documents[: settings.RERANK_TOP_N]

    parts = []
    for i, doc in enumerate(pruned_docs, 1):
        raw = doc.metadata.get("raw_content", doc.page_content) or ""
        cleaned = clean_chunk_text(raw)
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page_number", doc.metadata.get("page", "?"))
        score = doc.metadata.get("rerank_score")

        header = f"[Source {i}: {source}, Page {page}"
        if score is not None:
            header += f" [relevance: {score:.2f}]"
        header += "]"

        parts.append(f"{header}\n{cleaned}")

    context_str = "\n\n---\n\n".join(parts)
    return context_str, pruned_docs
