"""Drop boilerplate chunks (nav menus, footers, CTA banners) that appear
verbatim across many different scraped pages.

Real curriculum content is page-specific and should appear once (or a
handful of times at most across near-duplicate course pages). Site chrome
-- navigation, footers, "Book Now" banners, placement testimonials -- is
templated and repeats identically on every crawled page. Frequency is a
simple, content-agnostic signal for telling the two apart, without needing
a hardcoded list of marketing phrases that would go stale as the site changes.
"""
from collections import Counter


def drop_frequent_boilerplate(chunks: list[str], max_repeats: int = 3) -> list[str]:
    counts = Counter(chunks)
    return [c for c in chunks if counts[c] <= max_repeats]
