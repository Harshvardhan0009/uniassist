"""
Application configuration.

All settings are loaded from environment variables or a `.env` file.
Supports both local file persistence and shared Client-Server ChromaDB.
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and `.env`.
    
    Manages LLM providers (OpenRouter, xAI, OpenAI-compatible), Cohere reranking,
    HuggingFace sentence-transformer embeddings, and ChromaDB (Local or Server mode).
    """

    # ── Paths ────────────────────────────────────────────────────────
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "Data"
    CHROMA_PERSIST_DIR: Path = Path(__file__).resolve().parent.parent / "chroma_db"

    # ── API / Security ───────────────────────────────────────────────
    # Comma-separated list of CORS origins allowed to call the API.
    # Override in production, e.g. "https://uniassist.vercel.app".
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    # Bearer token required to trigger the /api/ingest endpoint.
    # If left empty, ingestion over HTTP is DISABLED (the CLI pipeline still works).
    INGEST_TOKEN: str = ""
    # Per-IP requests/minute for /api/query. 0 disables rate limiting.
    RATE_LIMIT_PER_MINUTE: int = 0

    # ── ChromaDB (Local or Client-Server Mode) ─────────────────────────
    # If CHROMA_HOST is empty, runs in local file mode using CHROMA_PERSIST_DIR.
    # If CHROMA_HOST is set (e.g. 'my-chroma.onrender.com' or 'localhost'), connects via HttpClient.
    CHROMA_HOST: str = ""
    CHROMA_PORT: int = 8001
    CHROMA_SSL: bool = False
    CHROMA_AUTH_TOKEN: str = ""
    CHROMA_COLLECTION: str = "university_docs"

    # ── Embedding ────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Max concurrent LLM calls when summarizing chunks during ingestion.
    SUMMARY_CONCURRENCY: int = 5

    # ── LLM (OpenRouter / xAI Grok / OpenAI-compatible) ───────────────
    # Provider-neutral names. Legacy GROK_* env vars are still accepted.
    LLM_API_KEY: str = Field(
        default="", validation_alias=AliasChoices("LLM_API_KEY", "GROK_API_KEY")
    )
    LLM_BASE_URL: str = Field(
        default="https://api.x.ai/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "GROK_BASE_URL"),
    )
    LLM_MODEL: str = Field(
        default="grok-3-mini",
        validation_alias=AliasChoices("LLM_MODEL", "GROK_MODEL"),
    )

    # ── Cohere Reranker ──────────────────────────────────────────────
    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-v3.5"

    # ── Retrieval tuning ─────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 20  # broad candidate set from ChromaDB
    RERANK_TOP_N: int = 5     # precise set after Cohere rerank
    # Drop candidates whose cosine relevance score is below this threshold.
    # 0.0 keeps everything; raise it (e.g. 0.2) to filter weak matches.
    RETRIEVAL_MIN_SCORE: float = 0.0

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ── Helpers ──────────────────────────────────────────────────────

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse ALLOWED_ORIGINS into a clean list of origins."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def is_chroma_server(self) -> bool:
        """True if configured to connect to a remote/shared ChromaDB server."""
        return bool(self.CHROMA_HOST and self.CHROMA_HOST.strip())

    @property
    def has_llm(self) -> bool:
        return bool(self.LLM_API_KEY)

    @property
    def has_cohere(self) -> bool:
        return bool(self.COHERE_API_KEY)


settings = Settings()
