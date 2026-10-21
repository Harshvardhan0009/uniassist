"""
Tests for /api/query and /api/query/stream request validation.
"""


def test_query_rejects_too_short(client):
    """POST /api/query rejects questions shorter than 3 characters with 422."""
    response = client.post("/api/query", json={"question": "hi"})
    assert response.status_code == 422

    detail = response.json().get("detail", [])
    assert any("question" in str(err) for err in detail)


def test_query_rejects_too_long(client):
    """POST /api/query rejects questions longer than 1000 characters with 422."""
    long_question = "What is the dress code? " * 50  # >1000 chars
    assert len(long_question) > 1000

    response = client.post("/api/query", json={"question": long_question})
    assert response.status_code == 422

    detail = response.json().get("detail", [])
    assert any("question" in str(err) for err in detail)


def test_query_rejects_empty(client):
    """POST /api/query rejects missing/empty question with 422."""
    response = client.post("/api/query", json={"question": ""})
    assert response.status_code == 422


def test_stream_query_rejects_too_short(client):
    """POST /api/query/stream also validates question length."""
    response = client.post("/api/query/stream", json={"question": "no"})
    assert response.status_code == 422
