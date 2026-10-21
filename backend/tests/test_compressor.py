"""
Tests for context compression, adaptive tail pruning, and whitespace cleanup.
"""

from langchain_core.documents import Document
from app.query.compressor import prune_candidate_chunks, clean_chunk_text


def test_compressor_prunes_tail_chunks_on_synthetic_scores():
    """
    Compressor preserves high-scoring top chunks and prunes low-scoring tail chunks.
    With top score = 0.95:
      - doc 1 (0.95): kept (min_chunks)
      - doc 2 (0.90): kept (min_chunks)
      - doc 3 (0.85): kept (score >= 0.20 and score >= 0.35 * 0.95)
      - doc 4 (0.15): dropped (score < 0.20)
      - doc 5 (0.05): dropped (score < 0.20)
    """
    scores = [0.95, 0.90, 0.85, 0.15, 0.05]
    docs = [
        Document(
            page_content=f"Content chunk {i}",
            metadata={"rerank_score": score, "source_file": "doc.pdf"},
        )
        for i, score in enumerate(scores)
    ]

    pruned = prune_candidate_chunks(docs, min_rel_ratio=0.35, min_abs_score=0.20, min_chunks=2)
    assert len(pruned) == 3
    assert [d.metadata["rerank_score"] for d in pruned] == [0.95, 0.90, 0.85]


def test_compressor_keeps_min_chunks_even_on_low_scores():
    """Compressor guarantees at least min_chunks are preserved even if all scores are low."""
    scores = [0.10, 0.08, 0.05, 0.02]
    docs = [
        Document(
            page_content=f"Low content {i}",
            metadata={"rerank_score": score, "source_file": "doc.pdf"},
        )
        for i, score in enumerate(scores)
    ]

    pruned = prune_candidate_chunks(docs, min_chunks=2)
    assert len(pruned) == 2
    assert [d.metadata["rerank_score"] for d in pruned] == [0.10, 0.08]


def test_clean_chunk_text_whitespace_and_empty_tables():
    """clean_chunk_text collapses excessive newlines and strips empty table divider rows."""
    raw = (
        "Heading\n\n\n\n"
        "Paragraph line with trailing spaces   \r\n"
        "|  |  |  |\n"
        "| Valid | Row |\n"
    )
    cleaned = clean_chunk_text(raw)

    assert "\n\n\n" not in cleaned
    assert "|  |  |  |" not in cleaned
    assert "| Valid | Row |" in cleaned
    assert "Paragraph line with trailing spaces" in cleaned
