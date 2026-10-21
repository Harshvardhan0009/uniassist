"""
Tests for chunker splitting behavior and metadata enrichment.
"""

from langchain_core.documents import Document
from app.ingestion.chunker import chunk_documents, CHUNK_SIZE, CHUNK_OVERLAP


def test_chunker_keeps_short_document_whole():
    """Documents under CHUNK_SIZE (1500 chars) are kept intact as a single chunk."""
    content = "This is a short university policy document."
    doc = Document(page_content=content, metadata={"source": "test.pdf", "page": 1})

    chunks = chunk_documents([doc], source_file="test.pdf")
    assert len(chunks) == 1
    assert chunks[0].page_content == content
    assert chunks[0].metadata["source_file"] == "test.pdf"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_type"] == "text_only"


def test_chunker_splits_long_document_within_size():
    """Documents exceeding CHUNK_SIZE are split into chunks <= 1500 chars."""
    paragraphs = [
        f"Section {i}: Lovely Professional University policy guidelines on student regulations and academic criteria. " * 5
        for i in range(15)
    ]
    long_content = "\n\n".join(paragraphs)
    assert len(long_content) > 3000

    doc = Document(page_content=long_content, metadata={"source": "policy.pdf", "page": 2})
    chunks = chunk_documents([doc], source_file="policy.pdf")

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE

    # Check indices are sequential and 0-indexed page is converted to 1-indexed page_number
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["source_file"] == "policy.pdf"
        assert chunk.metadata["page_number"] == 3  # input page 2 + 1


def test_chunker_overlap():
    """Adjacent chunks should share overlap content."""
    words = [f"word{i}" for i in range(500)]
    content = " ".join(words)
    doc = Document(page_content=content, metadata={"source": "test.txt", "page": 1})

    chunks = chunk_documents([doc], source_file="test.txt")
    assert len(chunks) >= 2

    # Verify that the end of chunk 0 appears in chunk 1
    tail_of_first = chunks[0].page_content[-100:]
    assert any(token in chunks[1].page_content for token in tail_of_first.split() if len(token) > 4)
