"""Drop chunks that repeat verbatim across many scraped pages -- nav menus,
footers, and CTA banners are templated site chrome, while real curriculum
text is page-specific. Frequency-based, so no hardcoded phrase list to
maintain as the site's copy changes.
"""
from collections import Counter


def drop_frequent_boilerplate(chunks: list[str], max_repeats: int = 3) -> list[str]:
    counts = Counter(chunks)
    return [c for c in chunks if counts[c] <= max_repeats]
