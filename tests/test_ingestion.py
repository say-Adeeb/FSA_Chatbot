"""Unit tests for cleaner and chunker (no network required)."""
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import split_text


class TestCleaner:
    def test_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_collapses_whitespace(self):
        assert clean_text("hello    world\n\n\ntest") == "hello world test"

    def test_strips_non_ascii_symbols(self):
        assert "★" not in clean_text("data ★ science")

    def test_drops_single_char_tokens(self):
        # single-character tokens are removed by design
        assert clean_text("a data b science") == "data science"


class TestChunker:
    def test_returns_chunks(self):
        text = "Data Science. " * 200
        chunks = split_text(text)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_short_text_single_chunk(self):
        chunks = split_text("short text")
        assert chunks == ["short text"]
