"""Unit tests for the boilerplate-frequency filter."""
from app.ingestion.dedup import drop_frequent_boilerplate


def test_drops_chunks_repeated_beyond_threshold():
    chunks = ["nav menu"] * 5 + ["real curriculum content"] * 1
    result = drop_frequent_boilerplate(chunks, max_repeats=3)
    assert "nav menu" not in result
    assert result.count("real curriculum content") == 1


def test_keeps_chunks_at_or_under_threshold():
    chunks = ["intro appears on two course pages"] * 2
    result = drop_frequent_boilerplate(chunks, max_repeats=3)
    assert result.count("intro appears on two course pages") == 2


def test_empty_input():
    assert drop_frequent_boilerplate([]) == []
