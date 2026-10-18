"""Query Rewriter module (Phase 15).

Reformulates conversational follow-up questions containing pronouns or ambiguous references
into standalone, self-contained search queries using conversation history.

Used during retrieval to resolve context drift in multi-turn dialogues.
"""

from __future__ import annotations

import logging
import re
import time

from . import paths

paths.ensure_backend_on_path()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger("eval.rewriter")

REWRITE_SYSTEM_PROMPT = (
    "You are a search query reformulation expert for a university information assistant. "
    "Given a conversation history between a student and the assistant, and a new follow-up question, "
    "rephrase the follow-up question to be an independent, self-contained search query.\n\n"
    "Rules:\n"
    "1. Replace pronouns (it, that, they, this, there) with the specific entity, policy, or subject being discussed.\n"
    "2. Preserve essential domain keywords (e.g. Library Policy, Career Services, OJT/Internship, Semester Abroad, Academic Benefits, Campus Map).\n"
    "3. Keep the query concise, factual, and targeted for semantic document search.\n"
    "4. Do NOT answer the question. Output ONLY the rewritten query text and nothing else."
)


def _to_history_text(history: list[dict] | None) -> str:
    """Format history list into a clean transcript string."""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _clean_query(text: str) -> str:
    """Strip extraneous quotation marks, reasoning tags, or preambles."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    text = re.sub(r"<think>[\s\S]*$", "", text).strip()
    text = text.replace("<think>", "").replace("</think>", "").strip()
    text = text.strip("\"' \n\t")
    # Strip common preamble if any
    for prefix in [
        "Rewritten query:",
        "Standalone query:",
        "Search query:",
        "Rewritten question:",
    ]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip(" \"'")
    return text


def rewrite_query(
    question: str,
    history: list[dict] | None = None,
    primary_api_key: str | None = None,
    primary_base_url: str | None = None,
    primary_model: str | None = None,
) -> tuple[str, float]:
    """Rewrite a conversational question into a standalone query.

    Returns:
        (rewritten_query, latency_ms)
    """
    if not history:
        return question, 0.0

    t0 = time.perf_counter()
    history_str = _to_history_text(history)
    user_prompt = f"Chat History:\n{history_str}\n\nFollow-up Question: {question}\n\nStandalone Search Query:"

    api_key = primary_api_key or settings.LLM_API_KEY
    base_url = primary_base_url or settings.LLM_BASE_URL
    model = primary_model or settings.LLM_MODEL

    # Try primary LLM first
    if api_key:
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.0,
                max_tokens=512,
            )
            messages = [
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            resp = llm.invoke(messages)
            raw_text = resp.content if isinstance(resp.content, str) else str(resp.content)
            rewritten = _clean_query(raw_text)
            if rewritten:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return rewritten, latency_ms
        except Exception as e:
            logger.warning(f"Primary LLM query rewriting failed: {e}. Trying fallback.")

    # Try fallback LLM (Groq)
    fallback_key = getattr(settings, "FALLBACK_LLM_API_KEY", "")
    fallback_url = getattr(settings, "FALLBACK_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    fallback_model = getattr(settings, "FALLBACK_LLM_MODEL", "openai/gpt-oss-120b")

    if fallback_key:
        try:
            llm = ChatOpenAI(
                api_key=fallback_key,
                base_url=fallback_url,
                model=fallback_model,
                temperature=0.0,
                max_tokens=512,
            )
            messages = [
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            resp = llm.invoke(messages)
            raw_text = resp.content if isinstance(resp.content, str) else str(resp.content)
            rewritten = _clean_query(raw_text)
            if rewritten:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                return rewritten, latency_ms
        except Exception as e:
            logger.warning(f"Fallback LLM query rewriting failed: {e}.")

    # If all LLMs fail, fall back safely to original question
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return question, latency_ms
