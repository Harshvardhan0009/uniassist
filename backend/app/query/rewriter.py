"""
Query Rewriter module (Phase 15).

Reformulates conversational follow-up questions containing pronouns or ambiguous references
into standalone, self-contained search queries using conversation history.
"""

from __future__ import annotations

import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

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

# Ambiguous pronouns or conversational follow-up triggers
PRONOUN_PATTERN = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|there|here|what about|how about|and the|also|same)\b",
    re.IGNORECASE,
)


def should_rewrite(question: str, history: list[dict] | None) -> bool:
    """Determine whether a question requires contextual reformulation."""
    if not history or not settings.ENABLE_QUERY_REWRITING:
        return False
    # If the user asks a very short question or uses pronouns/follow-up markers
    q = question.strip()
    if len(q.split()) <= 4:
        return True
    if PRONOUN_PATTERN.search(q):
        return True
    return False


def _to_history_text(history: list[dict] | None) -> str:
    """Format history list into a clean transcript string."""
    if not history:
        return ""
    lines = []
    for turn in history[-4:]:  # Keep up to last 4 turns
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
) -> tuple[str, float]:
    """Rewrite a conversational question into a standalone query.

    Returns:
        (rewritten_query, latency_ms)
    """
    if not should_rewrite(question, history):
        return question, 0.0

    t0 = time.perf_counter()
    history_str = _to_history_text(history)
    user_prompt = f"Chat History:\n{history_str}\n\nFollow-up Question: {question}\n\nStandalone Search Query:"

    providers = []
    if settings.has_llm:
        providers.append(("primary", settings.LLM_API_KEY, settings.LLM_BASE_URL, settings.LLM_MODEL))
    if settings.has_fallback_llm:
        providers.append(("fallback", settings.FALLBACK_LLM_API_KEY, settings.FALLBACK_LLM_BASE_URL, settings.FALLBACK_LLM_MODEL))

    for role, api_key, base_url, model in providers:
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.0,
                max_tokens=256,
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
                logger.info("Rewrote conversational query '%s' → '%s' (%s, %.1f ms)", question, rewritten, role, latency_ms)
                return rewritten, latency_ms
        except Exception as e:
            logger.warning("%s LLM query rewriting failed: %s", role, e)
            continue

    # Fallback safely to original question
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return question, latency_ms
