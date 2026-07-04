"""Unit tests for vectordb search logic with embeddings mocked (no model download)."""
import numpy as np
import pytest

import app.retrieval.vectordb as vdb


@pytest.fixture
def populated_db(monkeypatch):
    """Build a tiny in-memory index with deterministic fake embeddings."""
    docs = [
        "Data Science curriculum: statistics and machine learning",
        "Data Analyst course: SQL and Power BI",
        "Artificial Intelligence: neural networks and NLP",
    ]

    # Fake embeddings: one-hot-ish vectors so search is deterministic.
    def fake_embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            v = np.zeros(vdb.settings.EMBEDDING_DIM, dtype="float32")
            if "data science" in t.lower() or "statistics" in t.lower():
                v[0] = 1.0
            elif "analyst" in t.lower() or "sql" in t.lower():
                v[1] = 1.0
            else:
                v[2] = 1.0
            out.append(v)
        return np.array(out, dtype="float32")

    monkeypatch.setattr(vdb, "embed_texts", fake_embed)
    # Reset global state
    import faiss
    vdb.index = faiss.IndexFlatIP(vdb.settings.EMBEDDING_DIM)
    vdb.documents = []
    vdb.bm25 = None
    vdb.add_documents(docs)
    return vdb


class TestSearch:
    def test_empty_db_returns_empty(self, monkeypatch):
        monkeypatch.setattr(vdb, "documents", [])
        assert vdb.search_documents("anything") == []

    def test_returns_results(self, populated_db):
        results = populated_db.search_documents("statistics machine learning", k=3)
        assert len(results) >= 1

    def test_course_boost_orders_matches_first(self, populated_db):
        results = populated_db.search_documents("course", k=3, course="Data Analyst")
        assert "Analyst" in results[0]

    def test_boost_terms_order_matches_first_without_course(self, populated_db):
        # Generic intent boost (location/contact/etc.) works the same way as
        # the course boost, but without requiring a detected course.
        results = populated_db.search_documents("course", k=3, boost_terms=["sql"])
        assert "SQL" in results[0]

    def test_dim_mismatch_raises(self, populated_db, monkeypatch):
        def wrong_dim(texts):
            n = 1 if isinstance(texts, str) else len(texts)
            return np.zeros((n, 999), dtype="float32")
        monkeypatch.setattr(vdb, "embed_texts", wrong_dim)
        with pytest.raises(ValueError):
            populated_db.add_documents(["new doc"])
