"""API tests using FastAPI TestClient with the RAG pipeline mocked."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Provide a dummy key so settings.validate() passes during lifespan startup,
    # and stop load_index from doing real work.
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import app.core.config as cfg
    cfg.settings.GROQ_API_KEY = "test-key"

    with patch("app.main.load_index"):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_home(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "running" in r.json()["message"]


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_chat_status(client):
    assert client.get("/chat/").json()["status"] == "ok"


@patch("app.api.chat.ask_rag", return_value="Here is the Data Science curriculum.")
def test_chat_post_success(_mock, client):
    r = client.post("/chat/", json={"message": "What is in Data Science?"})
    assert r.status_code == 200
    assert r.json()["reply"] == "Here is the Data Science curriculum."


def test_chat_post_empty_message_rejected(client):
    # min_length=1 validation should reject empty string
    r = client.post("/chat/", json={"message": ""})
    assert r.status_code == 422


@patch("app.api.chat.ask_rag", side_effect=RuntimeError("boom"))
def test_chat_post_handles_internal_error(_mock, client):
    r = client.post("/chat/", json={"message": "hello"})
    assert r.status_code == 500
    assert "Failed" in r.json()["detail"]


@patch("app.api.chat.ask_rag", return_value="Here is the Data Science curriculum.")
def test_chat_post_generates_session_id_when_absent(_mock, client):
    r = client.post("/chat/", json={"message": "What is in Data Science?"})
    assert r.status_code == 200
    assert r.json()["session_id"]


@patch("app.api.chat.ask_rag", return_value="Mocked answer")
def test_chat_post_echoes_provided_session_id(_mock, client):
    r = client.post("/chat/", json={"message": "hi", "session_id": "my-session-123"})
    assert r.status_code == 200
    assert r.json()["session_id"] == "my-session-123"


@patch("app.api.chat.ask_rag", return_value="ok")
def test_chat_post_rate_limited(_mock, client, monkeypatch):
    import app.api.chat as chat_module
    from app.core.rate_limiter import RateLimiter

    monkeypatch.setattr(chat_module, "_rate_limiter", RateLimiter(max_requests=2, window_seconds=60))

    assert client.post("/chat/", json={"message": "hi"}).status_code == 200
    assert client.post("/chat/", json={"message": "hi"}).status_code == 200
    r = client.post("/chat/", json={"message": "hi"})
    assert r.status_code == 429
