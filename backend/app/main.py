"""
UniAssist — FastAPI Backend

Provides REST API endpoints for:
  - POST /api/query     → Ask a question against the university document corpus
  - POST /api/ingest    → Trigger document ingestion pipeline
  - GET  /api/health    → Health check
"""

import logging
import secrets
import sys
import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from app.config import settings

# Ensure UTF-8 stdout/stderr so rich/logging can emit Unicode (→, ✓, ⚠, —)
# on Windows consoles that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title="UniAssist API",
    description="University RAG system — ask questions about university documents",
    version="0.1.0",
)

# CORS — allow the configured frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security ─────────────────────────────────────────────────────────


def verify_ingest_token(authorization: str | None = Header(default=None)) -> None:
    """
    Guard the ingestion endpoint with a bearer token.

    Ingestion is disabled over HTTP unless INGEST_TOKEN is configured.
    """
    if not settings.INGEST_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Ingestion via the API is disabled. Set INGEST_TOKEN to enable it.",
        )
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, settings.INGEST_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing ingestion token.")


# Best-effort in-memory per-IP rate limiter (per process). For multi-worker
# deployments use a shared store (e.g. Redis) instead.
_request_log: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    """Throttle requests per client IP when RATE_LIMIT_PER_MINUTE > 0."""
    limit = settings.RATE_LIMIT_PER_MINUTE
    if limit <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _request_log[ip]
    while bucket and now - bucket[0] > 60.0:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    bucket.append(now)


# ── Request/Response Models ──────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=1000, description="The question to ask"
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Prior conversation turns for follow-up context",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    has_llm: bool
    timing: dict
    candidates_retrieved: int
    candidates_reranked: int


class IngestResponse(BaseModel):
    status: str
    pdfs_processed: int = 0
    total_elements: int = 0
    total_chunks: int = 0
    documents_stored: int = 0


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    cohere_configured: bool
    embedding_model: str
    collection: str
    chroma_mode: str
    chroma_host: str | None = None
    chroma_connected: bool
    documents_indexed: int | None = None


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check system health, configuration, and ChromaDB connectivity."""
    from app.ingestion.embedder import check_chroma

    chroma = check_chroma()
    return HealthResponse(
        status="ok" if chroma["connected"] else "degraded",
        llm_configured=settings.has_llm,
        cohere_configured=settings.has_cohere,
        embedding_model=settings.EMBEDDING_MODEL,
        collection=settings.CHROMA_COLLECTION,
        chroma_mode="client_server" if settings.is_chroma_server else "local_persistent",
        chroma_host=f"{'https' if settings.CHROMA_SSL else 'http'}://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}" if settings.is_chroma_server else None,
        chroma_connected=chroma["connected"],
        documents_indexed=chroma["documents"],
    )


@app.post("/api/query", response_model=QueryResponse, dependencies=[Depends(rate_limit)])
async def ask_question(request: QueryRequest):
    """
    Ask a question against the university document corpus.

    The pipeline: embed query → retrieve from ChromaDB → rerank → generate answer.
    """
    from app.query.chain import query

    try:
        result = query(request.question, history=[m.model_dump() for m in request.history])
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process the query. Please try again later.",
        )


@app.post(
    "/api/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_ingest_token)],
)
async def run_ingestion():
    """
    Trigger the document ingestion pipeline.

    Protected by a bearer token (INGEST_TOKEN). Processes all documents in the
    data directory, chunks them, generates summaries (if an LLM is configured),
    embeds, and stores them in ChromaDB.
    """
    from app.ingestion.pipeline import run_pipeline

    try:
        result = run_pipeline()
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Ingestion failed. Check the server logs for details.",
        )
