from pydantic import BaseModel, Field

from app.core.config import settings


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=settings.MAX_MESSAGE_LENGTH)
    # Omit on the first message; echo back the session_id from the response
    # on subsequent requests so the bot can remember the last discussed
    # course and answer short follow-ups ("what about fees?") in context.
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
