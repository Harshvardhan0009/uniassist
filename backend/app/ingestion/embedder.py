"""
Embedding & Vector Storage — Step 6 of the pipeline.

Embeds summarized chunks using HuggingFace embeddings (all-MiniLM-L6-v2)
via LangChain and stores them in ChromaDB.

Supports both:
  1. Client-Server Mode: Shared remote ChromaDB over HTTP (Render / Railway / EC2)
  2. Local File Mode: Embedded ChromaDB in `chroma_db/` directory
"""

import logging
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


_embedding_fn = None
_vector_store = None

def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Create the embedding function used across ingestion and query.

    Uses `all-MiniLM-L6-v2` by default (~80MB, 384-dim).
    Swap to BGE-M3 later by changing EMBEDDING_MODEL in .env.
    """
    global _embedding_fn
    if _embedding_fn is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_fn = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_fn


def get_vector_store(
    embedding_function: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """
    Get a Chroma vector store instance (Client-Server or Local).

    Args:
        embedding_function: Optional — if not provided, creates one.

    Returns:
        A LangChain Chroma vector store.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if embedding_function is None:
        embedding_function = get_embedding_function()

    if settings.is_chroma_server:
        logger.info(
            f"Connecting to ChromaDB Server at "
            f"{'https' if settings.CHROMA_SSL else 'http'}://{settings.CHROMA_HOST}:{settings.CHROMA_PORT} "
            f"(collection: '{settings.CHROMA_COLLECTION}')"
        )
        headers = {}
        if settings.CHROMA_AUTH_TOKEN:
            headers["X-Chroma-Token"] = settings.CHROMA_AUTH_TOKEN

        client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            ssl=settings.CHROMA_SSL,
            headers=headers if headers else None,
        )

        _vector_store = Chroma(
            client=client,
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=embedding_function,
        )
        return _vector_store
    else:
        logger.info(
            f"Using local ChromaDB at '{settings.CHROMA_PERSIST_DIR}' "
            f"(collection: '{settings.CHROMA_COLLECTION}')"
        )
        _vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=embedding_function,
            persist_directory=str(settings.CHROMA_PERSIST_DIR),
        )
        return _vector_store


def embed_and_store(documents: list[Document]) -> int:
    """
    Embed documents and store them in ChromaDB.

    Args:
        documents: List of summarized Documents ready for embedding.

    Returns:
        Number of documents stored.
    """
    if not documents:
        logger.warning("No documents to embed.")
        return 0

    logger.info(f"Embedding {len(documents)} documents...")
    embedding_fn = get_embedding_function()
    vector_store = get_vector_store(embedding_fn)

    # ChromaDB metadata values must be str, int, float, or bool.
    sanitized_docs = _sanitize_metadata(documents)

    # Add documents in batches
    batch_size = 50
    total_stored = 0

    for i in range(0, len(sanitized_docs), batch_size):
        batch = sanitized_docs[i : i + batch_size]
        ids = [
            f"{doc.metadata.get('source_file', 'unknown')}__chunk_{doc.metadata.get('chunk_index', idx)}"
            for idx, doc in enumerate(batch)
        ]
        vector_store.add_documents(batch, ids=ids)
        total_stored += len(batch)
        logger.info(f"  Stored {total_stored}/{len(sanitized_docs)} documents")

    logger.info(
        f"✓ Embedding complete: {total_stored} documents in "
        f"collection '{settings.CHROMA_COLLECTION}'"
    )
    return total_stored


def _sanitize_metadata(documents: list[Document]) -> list[Document]:
    """Convert non-primitive metadata values to strings for ChromaDB."""
    sanitized = []
    for doc in documents:
        clean_meta = {}
        for key, value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
            elif isinstance(value, list):
                clean_meta[key] = str(value)
            elif value is None:
                clean_meta[key] = ""
            else:
                clean_meta[key] = str(value)
        sanitized.append(
            Document(page_content=doc.page_content, metadata=clean_meta)
        )
    return sanitized
