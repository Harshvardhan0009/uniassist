"""
Application configuration.

All settings are loaded from environment variables or a `.env` file.
Supports both local file persistence and shared Client-Server ChromaDB.
"""

from pathlib import Path
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

    # ── LLM (OpenRouter / Grok / OpenAI-compatible) ───────────────────
    GROK_API_KEY: str = ""
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    GROK_MODEL: str = "grok-3-mini"

    # ── Cohere Reranker ──────────────────────────────────────────────
    COHERE_API_KEY: str = ""
    COHERE_RERANK_MODEL: str = "rerank-v3.5"

    # ── Retrieval tuning ─────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 20  # broad candidate set from ChromaDB
    RERANK_TOP_N: int = 5     # precise set after Cohere rerank

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # ── Helpers ──────────────────────────────────────────────────────

    @property
    def is_chroma_server(self) -> bool:
        """True if configured to connect to a remote/shared ChromaDB server."""
        return bool(self.CHROMA_HOST and self.CHROMA_HOST.strip())

    @property
    def has_grok(self) -> bool:
        return bool(self.GROK_API_KEY)

    @property
    def has_cohere(self) -> bool:
        return bool(self.COHERE_API_KEY)


settings = Settings()
