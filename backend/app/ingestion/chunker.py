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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    is_separator_regex=False,
)


def chunk_documents(documents: list[Document], source_file: str = "") -> list[Document]:
    """
    Split page-level documents into retrieval-friendly chunks.
    Preserves table markdown and metadata.
    """
    if not documents:
        return []

    # If document already fits within chunk size, keep it whole
    chunks = []
    for doc in documents:
        if len(doc.page_content) <= CHUNK_SIZE:
            chunks.append(doc)
        else:
            sub_chunks = splitter.split_documents([doc])
            chunks.extend(sub_chunks)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = source_file or chunk.metadata.get("filename", "unknown")
        chunk.metadata["chunk_index"] = i
        if "page" in chunk.metadata and "page_number" not in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    logger.info(
        f"  → {len(chunks)} chunks from {source_file} "
        f"(avg {_avg_chunk_size(chunks)} chars/chunk)"
    )
    return chunks


# Alias for backward compatibility
chunk_by_title = chunk_documents


def _avg_chunk_size(chunks: list[Document]) -> int:
    """Average chunk size for logging."""
    if not chunks:
        return 0
    return sum(len(c.page_content) for c in chunks) // len(chunks)
