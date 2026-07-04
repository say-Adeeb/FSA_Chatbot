"""Unit tests for the in-memory rate limiter."""
from app.core.rate_limiter import RateLimiter


def test_allows_up_to_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("ip1")
    assert limiter.allow("ip1")
    assert limiter.allow("ip1")
    assert limiter.allow("ip1") is False


def test_keys_are_independent():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("ip1")
    assert limiter.allow("ip2")
    assert limiter.allow("ip1") is False


def test_window_expiry_allows_again(monkeypatch):
    import app.core.rate_limiter as rl

    fake_time = [0.0]
    monkeypatch.setattr(rl.time, "monotonic", lambda: fake_time[0])

    limiter = RateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("ip1")
    assert limiter.allow("ip1") is False

    fake_time[0] += 11
    assert limiter.allow("ip1") is True
