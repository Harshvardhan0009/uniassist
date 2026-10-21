"""
Pytest configuration and shared fixtures for offline smoke tests.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Enforce test environment variables before importing app
os.environ["CHROMA_HOST"] = ""
os.environ["RATE_LIMIT_PER_MINUTE"] = "0"
os.environ["INGEST_TOKEN"] = ""


@pytest.fixture(autouse=True)
def mock_embedding_function():
    """Prevent any test from initializing the real HuggingFace/E5 model or downloading weights."""
    from unittest.mock import MagicMock
    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [[0.0] * 768]
    mock_emb.embed_query.return_value = [0.0] * 768
    with patch("app.ingestion.embedder.get_embedding_function", return_value=mock_emb) as mock_fn:
        yield mock_fn


@pytest.fixture
def mock_chroma_healthy():
    """Mock embedder.check_chroma to simulate a connected local ChromaDB."""
    with patch("app.ingestion.embedder.check_chroma") as mock_check:
        mock_check.return_value = {
            "connected": True,
            "documents": 285,
            "mode": "local_persistent",
        }
        yield mock_check


@pytest.fixture
def client(mock_chroma_healthy):
    """FastAPI TestClient fixture with mocked external services."""
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
