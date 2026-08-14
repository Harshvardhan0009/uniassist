"""
Chunking — Step 4 of the pipeline.

Splits page-level documents into smaller, semantically coherent chunks
using LangChain's RecursiveCharacterTextSplitter. Each chunk retains
the source file and page metadata from the original document.

Since we're using PyPDFLoader (page-level extraction), we don't have
element-level categories. Instead we split pages into overlapping chunks
of ~1000 chars for good retrieval granularity.
"""

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Splitter configuration ───────────────────────────────────────────
CHUNK_SIZE = 1000        # chars per chunk
CHUNK_OVERLAP = 200      # overlap to preserve context across boundaries

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    is_separator_regex=False,
)


def chunk_documents(documents: list[Document], source_file: str = "") -> list[Document]:
    """
    Split page-level documents into smaller retrieval-friendly chunks.

    Each chunk Document has:
      - page_content: the chunk text
      - metadata.source_file: original filename
      - metadata.page_number: source page
      - metadata.chunk_type: always 'text_only' (PyPDF doesn't detect tables/images)

    Args:
        documents: List of page-level Documents from partitioning.
        source_file: Name of the source PDF.

    Returns:
        List of chunked Documents.
    """
    if not documents:
        return []

    # Split all pages into smaller chunks
    chunks = splitter.split_documents(documents)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = source_file or chunk.metadata.get("filename", "unknown")
        chunk.metadata["chunk_type"] = "text_only"
        chunk.metadata["chunk_index"] = i
        # Preserve page info
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1

    logger.info(
        f"  → {len(chunks)} chunks from {source_file} "
        f"(avg {_avg_chunk_size(chunks)} chars/chunk)"
    )
    return chunks


# Alias for backward compatibility with pipeline.py
chunk_by_title = chunk_documents


def _avg_chunk_size(chunks: list[Document]) -> int:
    """Average chunk size for logging."""
    if not chunks:
        return 0
    return sum(len(c.page_content) for c in chunks) // len(chunks)
