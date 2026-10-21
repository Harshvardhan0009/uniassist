"""
Tests for /api/health and /api/metrics endpoints.
"""


def test_health_returns_200_and_expected_keys(client):
    """GET /api/health returns 200 and validates against HealthResponse schema."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    expected_keys = {
        "status",
        "llm_configured",
        "cohere_configured",
        "embedding_model",
        "collection",
        "chroma_mode",
        "chroma_host",
        "chroma_connected",
        "documents_indexed",
    }
    assert expected_keys.issubset(data.keys())
    assert data["status"] == "ok"
    assert data["chroma_connected"] is True
    assert data["documents_indexed"] == 285
    assert data["embedding_model"] == "intfloat/e5-base-v2"


def test_metrics_returns_200_and_telemetry(client):
    """GET /api/metrics returns 200 and valid telemetry summary."""
    from app.telemetry import log_query_trace

    # Check initial empty state
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries_logged" in data

    # Log a synthetic trace and verify computed metrics
    log_query_trace(
        "What is the dress code?",
        {
            "answer": "Formal on Mondays and Fridays.",
            "sources": ["DressCode.pdf"],
            "has_llm": True,
            "model_used": "gemini-3.6-flash",
            "llm_role": "primary",
            "timing": {"retrieval": 0.05, "reranking": 0.35, "generation": 1.20, "total": 1.60},
            "candidates_retrieved": 20,
            "candidates_reranked": 5,
            "candidates_compressed": 3,
        },
    )

    response2 = client.get("/api/metrics")
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["recent_window_queries"] >= 1
    assert "latency_ms" in data2
    assert "p50_total" in data2["latency_ms"]
    assert "query_routing" in data2
