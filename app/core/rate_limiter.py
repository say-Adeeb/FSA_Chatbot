"""In-memory sliding-window rate limiter.

Best-effort only: per-process, not shared across workers/restarts. Sufficient
to blunt casual abuse of the paid Groq call behind /chat without adding an
external dependency (Redis, etc.). If this is ever deployed with multiple
worker processes, move the counters to a shared store instead.
"""
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window_seconds
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True
