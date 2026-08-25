"""
Answer Generator — Step 9 of the pipeline.

Pulls raw content from reranked documents' metadata and passes it
along with the user's query to the configured LLM for final answer generation.

Optimized for table comprehension, range matching (e.g. salary slabs, CGPA, stipends),
and structured responses.
"""

import json
import logging
import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Maximum number of prior conversation messages to feed the model.
MAX_HISTORY_MESSAGES = 6

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are LPUAssist, an intelligent AI assistant for Lovely Professional University (LPU) students and staff. "
            "Answer the user's question accurately and helpfully using ONLY the provided context from "
            "official university documents.\n\n"
            "Response Style:\n"
            "- Be CONCISE. Give direct, to-the-point answers. Avoid lengthy explanations or repeating the question.\n"
            "- Use short bullet points for lists. Keep total response under 150 words unless the question demands detail.\n"
            "- Do NOT include any thinking, reasoning, or internal monologue in your response.\n"
            "- Do NOT wrap your response in <think> tags or any XML-like tags.\n\n"
            "Critical Instructions for Tables, Numbers, and Ranges:\n"
            "1. Pay close attention to table columns, headers, and rows.\n"
            "2. When a user asks about a specific number, package, or stipend (e.g. '6 lakhs', '6 LPA', '15,000 stipend', '7.5 CGPA'):\n"
            "   - Look for range slabs in tables such as 'Above 5 & up to 10 LPA' (which includes 6 lakhs), 'Above 3 & up to 5 LPA', etc.\n"
            "   - Identify the exact category, eligible pathway courses, grade updation levels, and conditions mapped to that range.\n"
            "3. Format table-derived answers in clean, readable bullet points or structured summaries.\n"
            "4. If the answer is genuinely NOT in the context, state that clearly.\n"
            "5. Cite the source document(s) used.",
        ),
        MessagesPlaceholder("history", optional=True),
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
            raw = "\n".join(json.loads(raw))
        except ValueError:
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
    """
    if not documents:
        return "I couldn't find relevant information about that in the university documents."

    source_contents: dict[str, list[str]] = {}
    for doc in documents:
        source = doc.metadata.get("source_file", "Unknown").replace(".pdf", "")
        raw = _extract_raw(doc)
        if source not in source_contents:
            source_contents[source] = []
        source_contents[source].append(raw)

    answer_parts = [f"Based on the university documents, here's what I found:\n"]

    for source, contents in source_contents.items():
        combined = "\n".join(contents)
        combined = re.sub(r"\n{3,}", "\n\n", combined)
        combined = combined.strip()

        sentences = []
        for line in combined.split("\n"):
            line = line.strip()
            if len(line) > 15:
                sentences.append(line)

        if sentences:
            text = "\n".join(sentences)
            if len(text) > 900:
                cutoff = text[:900].rfind(".")
                if cutoff > 500:
                    text = text[: cutoff + 1]
                else:
                    text = text[:900] + "..."

            answer_parts.append(f"**From {source}:**\n{text}\n")

    return "\n".join(answer_parts)


def _to_history_messages(history: list[dict] | None) -> list:
    """Convert [{role, content}] turns into LangChain messages (capped)."""
    if not history:
        return []
    messages = []
    for turn in history:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        role = turn.get("role")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages[-MAX_HISTORY_MESSAGES:]


def _clean_answer_text(answer: str) -> str:
    """Strip leaked <think>…</think> reasoning blocks (and unclosed variants)."""
    answer = re.sub(r"<think>[\s\S]*?</think>", "", answer).strip()
    answer = re.sub(r"<think>[\s\S]*$", "", answer).strip()
    answer = answer.replace("<think>", "").replace("</think>", "").strip()
    return answer


def _invoke_llm(
    api_key: str, base_url: str, model: str, context: str, query: str, history: list[dict] | None
) -> str:
    """Call one OpenAI-compatible LLM and return a cleaned answer (raises on failure)."""
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,
        max_tokens=1024,
    )
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": context, "query": query, "history": _to_history_messages(history)}
    )
    return _clean_answer_text(answer)


def generate_answer(
    query: str, documents: list[Document], history: list[dict] | None = None
) -> dict:
    """
    Generate the final answer, with automatic LLM failover.

    Tries the **primary** LLM (``LLM_*``) first; on any error or empty response it
    falls back to the **fallback** LLM (``FALLBACK_LLM_*``, e.g. Groq); if both are
    unavailable/fail, returns a structured extract of the retrieved passages.

    Returns a dict with:
      - ``answer``: the generated (or extracted) text
      - ``sources``: list of source filenames
      - ``has_llm``: whether an LLM produced the answer
      - ``model_used`` / ``llm_role``: which model answered ("primary"/"fallback"), or None
    """
    sources = list(dict.fromkeys(doc.metadata.get("source_file", "Unknown") for doc in documents))
    context = _format_context(documents)

    # Ordered failover chain: primary first, then fallback (each only if configured).
    providers: list[tuple[str, str, str, str]] = []
    if settings.has_llm:
        providers.append(("primary", settings.LLM_API_KEY, settings.LLM_BASE_URL, settings.LLM_MODEL))
    if settings.has_fallback_llm:
        providers.append(
            ("fallback", settings.FALLBACK_LLM_API_KEY, settings.FALLBACK_LLM_BASE_URL, settings.FALLBACK_LLM_MODEL)
        )

    if not providers:
        logger.warning("No LLM configured — generating structured response from context.")
        return {"answer": _build_clean_answer(query, documents), "sources": sources,
                "has_llm": False, "model_used": None, "llm_role": None}

    last_error: Exception | None = None
    for role, api_key, base_url, model in providers:
        try:
            logger.info("Generating answer with %s LLM (%s)...", role, model)
            answer = _invoke_llm(api_key, base_url, model, context, query, history)
            if not answer:
                raise ValueError("empty answer from LLM (reasoning consumed the token budget?)")
            return {"answer": answer, "sources": sources, "has_llm": True,
                    "model_used": model, "llm_role": role}
        except Exception as e:  # noqa: BLE001 — try the next provider
            last_error = e
            logger.error("%s LLM (%s) failed: %s", role, model, str(e)[:200])
            continue

    logger.error("All LLM providers failed (%s); using extractive fallback.", last_error)
    return {"answer": _build_clean_answer(query, documents), "sources": sources,
            "has_llm": False, "model_used": None, "llm_role": None}
