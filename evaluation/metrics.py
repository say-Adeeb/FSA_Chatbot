"""Retrieval quality metrics.

Relevance is approximated by keyword containment (case-insensitive substring
match) since the corpus has no hand-labeled chunk IDs to compare against.
"""


def hit_rate_at_k(retrieved: list[str], keywords: list[str], k: int) -> int:
    """1 if any of the top-k retrieved chunks contains any keyword, else 0."""
    keywords_lc = [kw.lower() for kw in keywords]
    for doc in retrieved[:k]:
        doc_lc = doc.lower()
        if any(kw in doc_lc for kw in keywords_lc):
            return 1
    return 0


def reciprocal_rank(retrieved: list[str], keywords: list[str]) -> float:
    """1/rank of the first retrieved chunk containing any keyword, else 0."""
    keywords_lc = [kw.lower() for kw in keywords]
    for rank, doc in enumerate(retrieved, start=1):
        doc_lc = doc.lower()
        if any(kw in doc_lc for kw in keywords_lc):
            return 1.0 / rank
    return 0.0
