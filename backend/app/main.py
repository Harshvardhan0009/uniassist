"""
UniAssist — FastAPI Backend

Provides REST API endpoints for:
  - POST /api/query     → Ask a question against the university document corpus
  - POST /api/ingest    → Trigger document ingestion pipeline
  - GET  /api/health    → Health check
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.logging import RichHandler

from app.config import settings

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

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=1000, description="The question to ask"
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
    grok_configured: bool
    cohere_configured: bool
    embedding_model: str
    collection: str


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check system health and configuration status."""
    return HealthResponse(
        status="ok",
        grok_configured=settings.has_grok,
        cohere_configured=settings.has_cohere,
        embedding_model=settings.EMBEDDING_MODEL,
        collection=settings.CHROMA_COLLECTION,
    )


@app.post("/api/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question against the university document corpus.

    The pipeline: embed query → retrieve from ChromaDB → rerank → generate answer.
    """
    from app.query.chain import query

    try:
        result = query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/ingest", response_model=IngestResponse)
async def run_ingestion():
    """
    Trigger the document ingestion pipeline.

    Processes all PDFs in the data directory, chunks them,
    generates summaries (if LLM is configured), embeds, and stores in ChromaDB.
    """
    from app.ingestion.pipeline import run_pipeline

    try:
        result = run_pipeline()
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
