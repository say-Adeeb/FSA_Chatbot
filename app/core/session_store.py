"""In-memory conversation session store. Per-process and cleared on restart,
with LRU eviction so it can't grow unbounded. Tracks just enough short-term
context (last course discussed, a few recent exchanges) for follow-up
questions -- fine for a single instance, swap for a real store if that changes.
"""
import threading
from collections import OrderedDict, deque

MAX_SESSIONS = 500
MAX_TURNS_PER_SESSION = 4  # (question, answer) pairs kept per session


class SessionStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS, max_turns: int = MAX_TURNS_PER_SESSION) -> None:
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def _blank_session(self) -> dict:
        return {"history": deque(maxlen=self.max_turns), "last_course": None}

    def get(self, session_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return self._blank_session()
            self._sessions.move_to_end(session_id)
            return session

    def update(self, session_id: str, question: str, answer: str, course: str | None) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._blank_session()
                self._sessions[session_id] = session

            session["history"].append((question, answer))
            if course:
                session["last_course"] = course
            self._sessions.move_to_end(session_id)

            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)


session_store = SessionStore()
