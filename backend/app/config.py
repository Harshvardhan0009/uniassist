"""
Application configuration.

All settings are loaded from environment variables or a `.env` file.
API keys default to empty strings — the pipeline gracefully degrades:
  - No GROK_API_KEY  → summarization is skipped (raw text used), generation returns context only
  - No COHERE_API_KEY → reranking is skipped (raw retrieval order preserved)
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and `.env`.
    
    Manages LLM providers (xAI, OpenRouter, OpenAI-compatible), Cohere reranking,
    HuggingFace sentence-transformer embeddings, and persistent ChromaDB settings.
    """

    # ── Paths ────────────────────────────────────────────────────────
    DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "Data"
    CHROMA_PERSIST_DIR: Path = Path(__file__).resolve().parent.parent / "chroma_db"

    # ── Embedding ────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ─────────────────────────────────────────────────────
    CHROMA_COLLECTION: str = "university_docs"

    # ── Grok (xAI) — OpenAI-compatible endpoint ─────────────────────
    GROK_API_KEY: str = ""
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    GROK_MODEL: str = "grok-3-mini"

    # ── Cohere ───────────────────────────────────────────────────────
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
    def has_grok(self) -> bool:
        return bool(self.GROK_API_KEY)

    @property
    def has_cohere(self) -> bool:
        return bool(self.COHERE_API_KEY)


settings = Settings()
