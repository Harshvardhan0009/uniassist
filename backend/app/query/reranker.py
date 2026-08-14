"""
Reranker — Step 8 of the pipeline.

Uses Cohere's cross-encoder reranker (rerank-v3.5) via LangChain to
narrow down the broad candidate set to the most relevant documents.

If no COHERE_API_KEY is set, reranking is **skipped** — candidates
are returned in their original retrieval order (truncated to top_n).
"""

import logging

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """
    Rerank candidate documents using Cohere's cross-encoder.

    Args:
        query: The user's query.
        documents: Candidate documents from retrieval.
        top_n: Number of documents to keep after reranking.

    Returns:
        Top-n reranked Documents.
    """
    top_n = top_n or settings.RERANK_TOP_N

    if not documents:
        return []

    if not settings.has_cohere:
        logger.warning(
            "⚠ COHERE_API_KEY not set — skipping reranking. "
            "Using first %d retrieval results. "
            "Set COHERE_API_KEY in .env to enable reranking.",
            top_n,
        )
        return documents[:top_n]

    logger.info(
        f"Reranking {len(documents)} candidates → top-{top_n} "
        f"with {settings.COHERE_RERANK_MODEL}"
    )

    try:
        from langchain_cohere import CohereRerank

        reranker = CohereRerank(
            cohere_api_key=settings.COHERE_API_KEY,
            model=settings.COHERE_RERANK_MODEL,
            top_n=top_n,
        )

        # CohereRerank expects documents with page_content
        reranked = reranker.compress_documents(
            documents=documents,
            query=query,
        )

        logger.info(f"  Reranked to {len(reranked)} documents")
        return list(reranked)

    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to retrieval order.")
        return documents[:top_n]
