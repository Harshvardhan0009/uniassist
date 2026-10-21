"""
Tests for /api/ingest bearer token authorization.
"""

from unittest.mock import patch
from app.config import settings


def test_ingest_disabled_when_token_empty(client):
    """POST /api/ingest returns 403 when INGEST_TOKEN is not configured."""
    with patch.object(settings, "INGEST_TOKEN", ""):
        response = client.post("/api/ingest")
        assert response.status_code == 403
        assert "disabled" in response.json()["detail"].lower()


def test_ingest_unauthorized_when_token_missing(client):
    """POST /api/ingest returns 401 when Authorization header is missing."""
    with patch.object(settings, "INGEST_TOKEN", "valid-secret-token"):
        response = client.post("/api/ingest")
        assert response.status_code == 401
        assert "invalid or missing" in response.json()["detail"].lower()


def test_ingest_unauthorized_when_token_invalid(client):
    """POST /api/ingest returns 401 when Bearer token is incorrect."""
    with patch.object(settings, "INGEST_TOKEN", "valid-secret-token"):
        headers = {"Authorization": "Bearer wrong-token"}
        response = client.post("/api/ingest", headers=headers)
        assert response.status_code == 401
        assert "invalid or missing" in response.json()["detail"].lower()
