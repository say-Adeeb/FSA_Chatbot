"""Unit tests for the in-memory conversation session store."""
from app.core.session_store import SessionStore


def test_new_session_is_blank():
    store = SessionStore()
    session = store.get("unknown-id")
    assert session["last_course"] is None
    assert list(session["history"]) == []


def test_update_then_get_persists_state():
    store = SessionStore()
    store.update("s1", "What is in Data Science?", "Some answer", course="Data Science")
    session = store.get("s1")
    assert session["last_course"] == "Data Science"
    assert session["history"][-1] == ("What is in Data Science?", "Some answer")


def test_history_is_capped_per_session():
    store = SessionStore(max_turns=2)
    store.update("s1", "q1", "a1", course=None)
    store.update("s1", "q2", "a2", course=None)
    store.update("s1", "q3", "a3", course=None)
    session = store.get("s1")
    assert list(session["history"]) == [("q2", "a2"), ("q3", "a3")]


def test_evicts_oldest_session_over_capacity():
    store = SessionStore(max_sessions=2)
    store.update("s1", "q", "a", course=None)
    store.update("s2", "q", "a", course=None)
    store.update("s3", "q", "a", course=None)
    assert store.get("s1")["last_course"] is None
    assert list(store.get("s1")["history"]) == []
    assert list(store.get("s3")["history"]) == [("q", "a")]


def test_course_only_updates_when_provided():
    store = SessionStore()
    store.update("s1", "q1", "a1", course="Data Science")
    store.update("s1", "q2", "a2", course=None)
    assert store.get("s1")["last_course"] == "Data Science"
