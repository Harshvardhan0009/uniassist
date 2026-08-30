"""
Chunking — Step 4 of the pipeline.

Splits page-level documents into coherent chunks while preserving table structures
and context using LangChain's RecursiveCharacterTextSplitter.
"""

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Splitter configuration ───────────────────────────────────────────
# Larger chunk size so full tables (often 1000-2000 chars) remain together
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


# Default splitter used by production ingestion (2500/300).
splitter = _make_splitter(CHUNK_SIZE, CHUNK_OVERLAP)


def chunk_documents(
    documents: list[Document],
    source_file: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split page-level documents into retrieval-friendly chunks.
    Preserves table markdown and metadata.

    ``chunk_size`` / ``chunk_overlap`` default to the module constants
    (2500 / 300 — production behaviour). Pass explicit values to experiment with
    different chunking (used by the evaluation snapshot builder, Phase 10); the
    keep-whole threshold then follows the provided ``chunk_size``.
    """
    if not documents:
        return []

    size = CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    # Reuse the cached default splitter for production settings; build a local one otherwise.
    active_splitter = (
        splitter if (size == CHUNK_SIZE and overlap == CHUNK_OVERLAP) else _make_splitter(size, overlap)
    )

    # If document already fits within chunk size, keep it whole
    chunks = []
    for doc in documents:
        if len(doc.page_content) <= size:
            chunks.append(doc)
        else:
            sub_chunks = active_splitter.split_documents([doc])
            chunks.extend(sub_chunks)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = source_file or chunk.metadata.get("filename", "unknown")
        chunk.metadata["chunk_index"] = i
        category = chunk.metadata.get("category", "")
        chunk.metadata["chunk_type"] = "table" if category == "TableAndText" else "text_only"
        if "page" in chunk.metadata and "page_number" not in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    logger.info(
        f"  → {len(chunks)} chunks from {source_file} "
        f"(avg {_avg_chunk_size(chunks)} chars/chunk)"
    )
    return chunks


def _avg_chunk_size(chunks: list[Document]) -> int:
    """Average chunk size for logging."""
    if not chunks:
        return 0
    return sum(len(c.page_content) for c in chunks) // len(chunks)
