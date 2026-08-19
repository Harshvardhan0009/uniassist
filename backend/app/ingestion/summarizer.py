"""
Chunk Summarization — Step 5 of the pipeline.

Calls the configured LLM (via LangChain's ChatOpenAI-compatible client) to
generate a concise summary of each chunk. The summary becomes the searchable
`page_content`, while the raw text is preserved in metadata for accurate answer
generation.

If no LLM_API_KEY is set, summarization is **skipped** — raw text is
used as page_content directly. This lets you ingest and test retrieval
without an API key.
"""

import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a university document summarizer. Create a concise, "
            "search-friendly summary of the following content. Preserve key "
            "facts: dates, names, policy numbers, course codes, amounts. "
            "Keep it under 3 sentences.",
        ),
        ("human", "{content}"),
    ]
)


def _get_llm() -> ChatOpenAI | None:
    """Get the summarization LLM if an API key is configured."""
    if not settings.has_llm:
        return None
    return ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.0,
        max_tokens=256,
    )


def summarize_chunks(chunks: list[Document]) -> list[Document]:
    """
    Generate summaries for each chunk and restructure Documents.

    After summarization, each Document has:
      - page_content = summary (or raw text if no API key)
      - metadata.raw_content = original full text (used for generation)
      - metadata retains all other fields from chunking

    Args:
        chunks: List of chunked Documents from the chunker.

    Returns:
        List of Documents ready for embedding.
    """
    llm = _get_llm()

    if llm is None:
        logger.warning(
            "⚠ LLM_API_KEY not set — skipping summarization. "
            "Raw text will be used as page_content. "
            "Set LLM_API_KEY in .env to enable summarization."
        )
        return _passthrough(chunks)

    logger.info(f"Summarizing {len(chunks)} chunks with {settings.LLM_MODEL}...")
    chain = SUMMARY_PROMPT | llm

    # Summarize concurrently (bounded) instead of one blocking call per chunk.
    inputs = [{"content": chunk.page_content} for chunk in chunks]
    try:
        results = chain.batch(
            inputs,
            config={"max_concurrency": max(1, settings.SUMMARY_CONCURRENCY)},
            return_exceptions=True,
        )
    except Exception as e:
        logger.warning(f"Batch summarization failed ({e}); falling back to raw text.")
        results = [e] * len(chunks)

    summarized: list[Document] = []
    failures = 0
    for chunk, result in zip(chunks, results):
        if isinstance(result, BaseException):
            failures += 1
            summary = chunk.page_content  # fallback to raw
        else:
            summary = (getattr(result, "content", "") or "").strip() or chunk.page_content

        # Restructure: summary as searchable content, raw in metadata
        doc = Document(
            page_content=summary,
            metadata={
                **chunk.metadata,
                "raw_content": chunk.page_content,
            },
        )
        summarized.append(doc)

    if failures:
        logger.warning(f"  {failures}/{len(chunks)} chunk summaries fell back to raw text")
    logger.info(f"Summarization complete: {len(summarized)} chunks")
    return summarized


def _passthrough(chunks: list[Document]) -> list[Document]:
    """When no LLM is available, keep raw text as page_content but
    also store it in metadata.raw_content for consistency."""
    return [
        Document(
            page_content=chunk.page_content,
            metadata={
                **chunk.metadata,
                "raw_content": chunk.page_content,
            },
        )
        for chunk in chunks
    ]
