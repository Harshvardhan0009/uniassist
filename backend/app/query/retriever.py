"""
Retriever — Step 7 of the pipeline.

Embeds the user query and performs similarity search against ChromaDB
using the same embedding model used during ingestion.
"""

import logging

from langchain_core.documents import Document

from app.config import settings
from app.ingestion.embedder import get_vector_store

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """
    Retrieve candidate documents from ChromaDB via similarity search.

    Args:
        query: The user's natural language query.
        top_k: Number of candidates to retrieve. Defaults to RETRIEVAL_TOP_K.

    Returns:
        List of top-k candidate Documents with similarity scores in metadata.
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K

    logger.info(f"Retrieving top-{top_k} candidates for: '{query[:80]}...'")
    vector_store = get_vector_store()

    # similarity_search_with_relevance_scores returns (Document, score) tuples
    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
    )

    # Attach score to metadata for downstream use
    documents = []
    for doc, score in results:
        doc.metadata["relevance_score"] = round(score, 4)
        documents.append(doc)

    if documents:
        logger.info(
            f"  Retrieved {len(documents)} candidates "
            f"(score range: {documents[0].metadata['relevance_score']:.3f} "
            f"— {documents[-1].metadata['relevance_score']:.3f})"
        )
    else:
        logger.info("  No candidates found.")
    return documents
