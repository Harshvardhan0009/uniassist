"""Context Compression and Trimming module (Phase 19).

Provides deterministic, fact-preserving context compression:
1. Adaptive score-aware candidate pruning (drops irrelevant low-scoring tail chunks).
2. Structural whitespace and redundant markdown cleanup.
3. Compact, page-aware source header formatting.
"""

from __future__ import annotations

import re
from typing import Any
from langchain_core.documents import Document


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
    min_rel_ratio: float = 0.35,
    min_abs_score: float = 0.20,
    min_chunks: int = 2,
    max_chunks: int = 5,
) -> list[Document]:
    """Prune low-confidence tail chunks after cross-encoder reranking.

    Rules:
    1. Always keep up to `min_chunks` (default 2) if available.
    2. For ranks > min_chunks: keep only if score >= min_abs_score and score >= (min_rel_ratio * top_score).
    3. Never exceed `max_chunks`.
    """
    if not documents:
        return []

    top_doc = documents[0]
    top_score = float(top_doc.metadata.get("rerank_score", 1.0) or 1.0)

    # If the best chunk has a very low score (< 0.25), keep up to min_chunks to give the LLM enough context
    if top_score < 0.25:
        return documents[:min_chunks]

    kept: list[Document] = []
    for idx, doc in enumerate(documents, start=1):
        if idx <= min_chunks:
            kept.append(doc)
            continue

        score = float(doc.metadata.get("rerank_score", 0.0) or 0.0)
        # Check absolute and relative threshold
        if score >= min_abs_score and score >= (min_rel_ratio * top_score):
            kept.append(doc)

        if len(kept) >= max_chunks:
            break

    return kept


def format_compressed_context(
    documents: list[Document],
    enable_pruning: bool = True,
    min_rel_ratio: float = 0.35,
    min_abs_score: float = 0.20,
    min_chunks: int = 2,
    max_chunks: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Format documents into a compressed context prompt block with telemetry.

    Returns (context_text, stats_dict).
    """
    raw_chars_total = sum(len(doc.metadata.get("raw_content", doc.page_content) or "") for doc in documents)

    if enable_pruning:
        pruned_docs = prune_candidate_chunks(
            documents,
            min_rel_ratio=min_rel_ratio,
            min_abs_score=min_abs_score,
            min_chunks=min_chunks,
            max_chunks=max_chunks,
        )
    else:
        pruned_docs = documents[:max_chunks]

    parts = []
    compressed_chars_total = 0

    for i, doc in enumerate(pruned_docs, 1):
        raw_text = doc.metadata.get("raw_content", doc.page_content) or ""
        cleaned = clean_chunk_text(raw_text)
        compressed_chars_total += len(cleaned)

        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page_number")
        page_str = f", Page {page}" if page else ""
        score = doc.metadata.get("rerank_score")
        score_str = f" [relevance: {score:.2f}]" if score is not None else ""

        header = f"[Source {i}: {source}{page_str}{score_str}]"
        parts.append(f"{header}\n{cleaned}")

    context_str = "\n\n---\n\n".join(parts)

    savings_chars = max(0, raw_chars_total - len(context_str))
    savings_pct = (savings_chars / raw_chars_total * 100.0) if raw_chars_total > 0 else 0.0

    stats = {
        "original_chunks": len(documents),
        "kept_chunks": len(pruned_docs),
        "raw_chars": raw_chars_total,
        "compressed_chars": len(context_str),
        "chars_saved": savings_chars,
        "savings_pct": round(savings_pct, 2),
    }

    return context_str, stats
