"""Unit tests for the evaluation harness's retrieval metrics."""
from evaluation.metrics import hit_rate_at_k, reciprocal_rank


DOCS = [
    "Data Analyst course: SQL and Power BI",
    "SOC Analyst: phishing and incident response",
    "Artificial Intelligence: neural networks and NLP",
]


class TestHitRateAtK:
    def test_hit_when_keyword_present(self):
        assert hit_rate_at_k(DOCS, ["power bi"], k=3) == 1

    def test_miss_when_keyword_absent(self):
        assert hit_rate_at_k(DOCS, ["tableau"], k=3) == 0

    def test_respects_k_cutoff(self):
        assert hit_rate_at_k(DOCS, ["neural networks"], k=1) == 0
        assert hit_rate_at_k(DOCS, ["neural networks"], k=3) == 1

    def test_case_insensitive(self):
        assert hit_rate_at_k(DOCS, ["POWER BI"], k=3) == 1


class TestReciprocalRank:
    def test_first_rank_match(self):
        assert reciprocal_rank(DOCS, ["sql"]) == 1.0

    def test_second_rank_match(self):
        assert reciprocal_rank(DOCS, ["phishing"]) == 0.5

    def test_no_match_returns_zero(self):
        assert reciprocal_rank(DOCS, ["tableau"]) == 0.0
