"""In-memory conversation session store.

Best-effort only: per-process, not persisted, cleared on restart. Bounds
memory with LRU eviction so a long-running server can't accumulate unbounded
sessions or per-session history. Sufficient for tracking short-term context
(the last detected course, a couple of recent exchanges) without needing an
external store for a single-instance deployment.
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
