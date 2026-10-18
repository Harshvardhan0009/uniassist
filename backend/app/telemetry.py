"""
Observability & Tracing Telemetry (Phase 23).

Captures structured execution logs, latency waterfalls, token counts,
and pipeline telemetry for every RAG query.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("app.telemetry")

# In-memory sliding window of recent query traces (last 100 queries)
_RECENT_TRACES: deque[dict[str, Any]] = deque(maxlen=100)

# Optional append-only log file
_TELEMETRY_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_TELEMETRY_LOG_FILE = _TELEMETRY_LOG_DIR / "query_traces.jsonl"


def log_query_trace(
    question: str,
    result: dict[str, Any],
    client_ip: str | None = None,
) -> dict[str, Any]:
    """Record a structured telemetry event for a processed RAG query."""
    trace_id = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc).isoformat()

    timings = result.get("timing", {})
    # Convert timings from seconds to milliseconds for standard telemetry
    timings_ms = {
        k: round(v * 1000, 1) if isinstance(v, (int, float)) else v
        for k, v in timings.items()
    }

    trace = {
        "trace_id": trace_id,
        "timestamp": now_utc,
        "client_ip": client_ip or "internal",
        "question": question,
        "search_query": result.get("search_query"),
        "was_rewritten": bool(result.get("search_query")),
        "model_used": result.get("model_used"),
        "llm_role": result.get("llm_role"),
        "has_llm": result.get("has_llm", False),
        "candidates_retrieved": result.get("candidates_retrieved", 0),
        "candidates_reranked": result.get("candidates_reranked", 0),
        "candidates_compressed": result.get("candidates_compressed", 0),
        "sources": result.get("sources", []),
        "answer_length_chars": len(result.get("answer", "") or ""),
        "timings_ms": timings_ms,
    }

    _RECENT_TRACES.append(trace)

    # Persist to JSONL asynchronously / safely
    try:
        _TELEMETRY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _TELEMETRY_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Failed to append telemetry to file: %s", e)

    logger.info(
        "Trace [%s] total=%.0fms ret=%.0fms rerank=%.0fms gen=%.0fms role=%s",
        trace_id[:8],
        timings_ms.get("total", 0),
        timings_ms.get("retrieval", 0),
        timings_ms.get("reranking", 0),
        timings_ms.get("generation", 0),
        trace["llm_role"],
    )
    return trace


def get_telemetry_summary() -> dict[str, Any]:
    """Compute rolling metrics across recently recorded queries."""
    if not _RECENT_TRACES:
        return {
            "total_queries_logged": 0,
            "message": "No queries logged in current runtime session.",
        }

    total = len(_RECENT_TRACES)
    latencies = [t["timings_ms"].get("total", 0.0) for t in _RECENT_TRACES]
    gen_latencies = [t["timings_ms"].get("generation", 0.0) for t in _RECENT_TRACES]
    ret_latencies = [t["timings_ms"].get("retrieval", 0.0) for t in _RECENT_TRACES]
    rerank_latencies = [t["timings_ms"].get("reranking", 0.0) for t in _RECENT_TRACES]

    rewritten_count = sum(1 for t in _RECENT_TRACES if t["was_rewritten"])
    primary_count = sum(1 for t in _RECENT_TRACES if t.get("llm_role") == "primary")
    fallback_count = sum(1 for t in _RECENT_TRACES if t.get("llm_role") == "fallback")
    extractive_count = sum(1 for t in _RECENT_TRACES if not t.get("has_llm"))

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(total * 0.50)]
    p95 = latencies_sorted[int(total * 0.95)] if total >= 20 else latencies_sorted[-1]

    return {
        "recent_window_queries": total,
        "latency_ms": {
            "p50_total": round(p50, 1),
            "p95_total": round(p95, 1),
            "mean_total": round(sum(latencies) / total, 1),
            "mean_generation": round(sum(gen_latencies) / total, 1),
            "mean_retrieval": round(sum(ret_latencies) / total, 1),
            "mean_reranking": round(sum(rerank_latencies) / total, 1),
        },
        "query_routing": {
            "rewritten_ratio": round(rewritten_count / total, 3),
            "primary_model_ratio": round(primary_count / total, 3),
            "fallback_model_ratio": round(fallback_count / total, 3),
            "extractive_fallback_ratio": round(extractive_count / total, 3),
        },
        "latest_trace_id": _RECENT_TRACES[-1]["trace_id"],
    }
