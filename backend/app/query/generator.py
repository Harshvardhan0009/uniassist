"""
Answer Generator — Step 9 of the pipeline.

Pulls raw content from reranked documents' metadata and passes it
along with the user's query to Grok for final answer generation.

If no GROK_API_KEY is set, generates a clean, structured response
from the retrieved context (no raw dumps).
"""

import logging
import re
from ast import literal_eval

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are UniAssist, an intelligent AI assistant for Lovely Professional University (LPU) students and staff. "
            "Answer the user's question accurately and helpfully using ONLY the provided context from "
            "official university documents.\n\n"
            "Formatting & Tone Guidelines:\n"
            "- If the answer is present in the context, provide a clear, conversational, and well-structured response.\n"
            "- Use bullet points, bold section headers, and clean line breaks for readability.\n"
            "- If the answer is NOT in the context, honestly state that official documentation does not contain that info.\n"
            "- Include specific citations indicating which document or section was referenced.\n"
            "- Be direct, polite, concise, and helpful.",
        ),
        (
            "human",
            "Context from university documents:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Question: {query}",
        ),
    ]
)


def _extract_raw(doc: Document) -> str:
    """Get raw text content from a document, handling ChromaDB serialization."""
    raw = doc.metadata.get("raw_content", doc.page_content)
    if isinstance(raw, str) and raw.startswith("["):
        try:
            raw = "\n".join(literal_eval(raw))
        except (ValueError, SyntaxError):
            pass
    return raw.strip()


def _format_context(documents: list[Document]) -> str:
    """Build context string for LLM prompt."""
    parts = []
    for i, doc in enumerate(documents, 1):
        raw = _extract_raw(doc)
        source = doc.metadata.get("source_file", "Unknown")
        title = doc.metadata.get("title", "")
        header = f"[Source {i}: {source}"
        if title:
            header += f" — {title}"
        header += "]"
        parts.append(f"{header}\n{raw}")
    return "\n\n---\n\n".join(parts)


def _build_clean_answer(query: str, documents: list[Document]) -> str:
    """
    Build a clean, readable answer from retrieved documents WITHOUT an LLM.
    
    Instead of dumping raw text, this extracts key sentences that are most
    relevant and presents them in a structured format.
    """
    if not documents:
        return "I couldn't find relevant information about that in the university documents."

    # Gather content by source
    source_contents: dict[str, list[str]] = {}
    for doc in documents:
        source = doc.metadata.get("source_file", "Unknown").replace(".pdf", "")
        raw = _extract_raw(doc)
        if source not in source_contents:
            source_contents[source] = []
        source_contents[source].append(raw)

    # Build a structured response
    answer_parts = []
    answer_parts.append(f"Based on the university documents, here's what I found:\n")

    for source, contents in source_contents.items():
        combined = "\n".join(contents)

        # Clean up the text — remove excessive whitespace, page numbers, etc.
        combined = re.sub(r'\n{3,}', '\n\n', combined)
        combined = re.sub(r'^\s*\d+\s*$', '', combined, flags=re.MULTILINE)
        combined = combined.strip()

        # Extract meaningful sentences (skip very short lines)
        sentences = []
        for line in combined.split('\n'):
            line = line.strip()
            if len(line) > 20:  # Skip very short fragments
                sentences.append(line)

        if sentences:
            # Take the most relevant portion (first ~800 chars)
            text = '\n'.join(sentences)
            if len(text) > 800:
                # Try to cut at a sentence boundary
                cutoff = text[:800].rfind('.')
                if cutoff > 400:
                    text = text[:cutoff + 1]
                else:
                    text = text[:800] + "..."

            answer_parts.append(f"**From {source}:**\n{text}\n")

    answer_parts.append(
        "\n---\n*Note: For a more detailed and conversational answer, "
        "configure the GROK_API_KEY in the backend.*"
    )

    return "\n".join(answer_parts)


def generate_answer(query: str, documents: list[Document]) -> dict:
    """
    Generate the final answer using Grok LLM.

    Args:
        query: The user's question.
        documents: Reranked documents with context.

    Returns:
        Dict with:
          - answer: the generated text
          - sources: list of source filenames
          - has_llm: whether LLM was used
    """
    sources = list(
        {doc.metadata.get("source_file", "Unknown") for doc in documents}
    )

    if not settings.has_grok:
        logger.warning(
            "GROK_API_KEY not set — generating structured response from context."
        )
        return {
            "answer": _build_clean_answer(query, documents),
            "sources": sources,
            "has_llm": False,
        }

    logger.info(f"Generating answer with {settings.GROK_MODEL}...")
    context = _format_context(documents)

    llm = ChatOpenAI(
        api_key=settings.GROK_API_KEY,
        base_url=settings.GROK_BASE_URL,
        model=settings.GROK_MODEL,
        temperature=0.1,
        max_tokens=1024,
    )

    chain = ANSWER_PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke({"context": context, "query": query})
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        answer = _build_clean_answer(query, documents)

    return {
        "answer": answer,
        "sources": sources,
        "has_llm": True,
    }
