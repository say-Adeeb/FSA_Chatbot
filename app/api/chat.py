import logging
import uuid

from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.core.rate_limiter import RateLimiter
from app.models.chat_models import ChatRequest, ChatResponse
from app.retrieval.rag_pipeline import ask_rag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

_rate_limiter = RateLimiter(max_requests=settings.RATE_LIMIT_PER_MINUTE)


@router.get("/")
def chat_status():
    return {"status": "ok", "message": "Chat route is working"}


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request):
    client_ip = http_request.client.host if http_request.client else "unknown"
    if not _rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        answer = ask_rag(request.message, session_id=session_id)
    except Exception:
        logger.exception("Error handling chat request")
        raise HTTPException(status_code=500, detail="Failed to process the request.")
    return ChatResponse(reply=answer, session_id=session_id)
