"""RAG orchestration: detect course intent, retrieve context, generate answer."""
import random
import re

from app.core.config import settings
from app.core.session_store import session_store
from app.retrieval.vectordb import search_documents
from app.retrieval.llm import default_fallback_message, generate_answer

# Ordered longest-first so "artificial intelligence" wins over a bare "ai".
# Each entry maps aliases -> canonical course label.
COURSE_ALIASES = [
    (["artificial intelligence", "ai specialist", "ai course"], "Artificial Intelligence"),
    (["data science", "ds course"], "Data Science"),
    (["data analyst", "data analytics", "da course"], "Data Analyst"),
    (["soc analyst", "cyber security", "cybersecurity", "security operations"], "SOC Analyst"),
    (["devops"], "DevOps"),
    (["machine learning", "ml course"], "Machine Learning"),
]

# ---------------------------------------------------------------------------
# Dialogue-act short-circuits. None of these are "questions about the course
# catalog", so running them through retrieval+generation is guaranteed to
# fail grounding and burns a Groq call for nothing. Each phrase set is matched
# against the *entire* normalized message, so a real question that happens to
# start with one of these words (e.g. "hi, what does the AI course cover?")
# still goes through full RAG. Each category has multiple reply variants,
# chosen at random, so the bot doesn't repeat the identical sentence on every
# turn -- verbatim repetition is a big part of what makes a bot feel robotic.
# ---------------------------------------------------------------------------

GREETINGS = {
    "hi", "hii", "hiii", "hey", "heyy", "hello", "helo", "yo",
    "greetings", "good morning", "good afternoon", "good evening",
}
GREETING_RESPONSES = [
    "Hi! I'm the Full Stack Academy assistant. Ask me about our courses "
    "(Data Science, Data Analyst, SOC Analyst, Artificial Intelligence, and more), "
    "or about locations, contact details, or admissions.",
    "Hello! Happy to help -- ask me about any of our courses, fees, locations, or admissions.",
    "Hey there! I can tell you about our courses, branches, or how to get in touch. What would you like to know?",
]

ACKNOWLEDGMENTS = {
    "ok", "okay", "k", "kk", "alright", "cool", "got it", "gotit",
    "sounds good", "fine", "noted", "sure", "understood", "nice",
}
ACK_RESPONSES = [
    "Sounds good! Let me know if you have any other questions about our courses.",
    "Got it. Feel free to ask about courses, fees, locations, or admissions anytime.",
    "Alright! I'm here if you want to know more about anything else.",
]

THANKS = {
    "thanks", "thank you", "thankyou", "thanks a lot", "thank you so much",
    "appreciate it", "ty", "thx", "many thanks",
}
THANKS_RESPONSES = [
    "You're welcome! Let me know if there's anything else I can help with.",
    "Happy to help! Feel free to ask if you have more questions.",
    "Anytime! Reach out if you need anything else about our courses.",
]

FAREWELLS = {
    "bye", "goodbye", "bye bye", "see you", "see ya", "cya", "take care", "later",
}
FAREWELL_RESPONSES = [
    "Goodbye! Feel free to come back if you have more questions about our courses.",
    "Take care! Reach out anytime you want to know more about Full Stack Academy.",
    "Bye for now! Good luck, and let us know if you need anything else.",
]

NEEDS_CLARIFICATION = {
    "yes", "yeah", "yep", "yup", "no", "nope", "nah", "maybe", "not sure", "idk",
}
CLARIFICATION_RESPONSES = [
    "Could you tell me a bit more about what you'd like to know? I can help with "
    "courses, fees, locations, or admissions.",
    "I want to make sure I help with the right thing -- could you share a bit more detail?",
]

DIALOGUE_ACTS = [
    (GREETINGS, GREETING_RESPONSES),
    (ACKNOWLEDGMENTS, ACK_RESPONSES),
    (THANKS, THANKS_RESPONSES),
    (FAREWELLS, FAREWELL_RESPONSES),
    (NEEDS_CLARIFICATION, CLARIFICATION_RESPONSES),
]


def _normalize(question: str) -> str:
    return re.sub(r"[^a-z\s]", "", question.lower()).strip()


def _match_dialogue_act(question: str) -> str | None:
    """Return a canned reply if the *entire* message is smalltalk, else None."""
    normalized = _normalize(question)
    for phrases, responses in DIALOGUE_ACTS:
        if normalized in phrases:
            return random.choice(responses)
    return None


# Non-course FAQ intents: trigger words in the question -> marker substrings to
# boost in retrieved chunks. This does for general academy FAQs what the course
# boost already does for courses -- without it, a direct-answer chunk (e.g. the
# branch address list) can rank behind vaguer marketing prose that happens to
# share a query word, which makes the LLM under-weight the actual answer.
FAQ_BOOSTS = [
    (["location", "branch", "where is", "centre", "center", "address"],
     ["location", "branch", "road", "floor", "gachibowli", "tolichowki", "ameerpet", "charminar"]),
    (["contact", "phone", "email", "call", "number", "reach you"],
     ["contact", "phone", "+91", "email", "info@"]),
    (["fee", "fees", "cost", "price"],
     ["fee", "fees", "cost", "price"]),
    (["duration", "how long", "months", "weeks"],
     ["duration", "months", "weeks", "hours"]),
]

# Cues that a short question is a follow-up to the previous turn rather than
# a fresh, standalone one (e.g. "what about fees?" right after asking about
# Data Science should stay scoped to Data Science).
FOLLOWUP_HINTS = ["what about", "how about", "and the", "and what", "also", "what's the", "whats the"]


def _detect_boost_terms(question: str) -> list[str]:
    q = question.lower()
    terms: list[str] = []
    for triggers, markers in FAQ_BOOSTS:
        if any(t in q for t in triggers):
            terms.extend(markers)
    return terms


def _looks_like_followup(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in FOLLOWUP_HINTS) or len(question.split()) <= 5


def extract_course_name(question: str) -> str:
    """Detect a course using word-boundary matching (avoids 'ai' in 'training')."""
    q = question.lower()
    for aliases, canonical in COURSE_ALIASES:
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return canonical

    # Fallback: only capture "<name> course/program/bootcamp/curriculum/syllabus"
    # phrasing. A bare "in|for|about <phrase>" match is too permissive and
    # hallucinates a "course" out of unrelated questions (e.g. "fees for
    # working professionals" -> "Working Professionals").
    m = re.search(
        r"\b(?:in|for|about)\s+(?:the\s+)?([a-zA-Z][a-zA-Z ]{2,40}?)\s+"
        r"(?:course|program|bootcamp|curriculum|syllabus)\b",
        question,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().title()
    return "the course"


def ask_rag(question: str, session_id: str | None = None) -> str:
    if not question or not question.strip():
        return "Please ask a specific question."

    canned = _match_dialogue_act(question)
    if canned is not None:
        return canned

    session = session_store.get(session_id) if session_id else None

    course = extract_course_name(question)

    # Carry the last discussed course forward for short follow-up questions
    # ("what about fees?" right after a Data Science question) instead of
    # treating every message as a fresh, context-free query.
    if course == "the course" and session and session["last_course"] and _looks_like_followup(question):
        course = session["last_course"]

    # Only bias the query toward a course if one was confidently detected.
    if course != "the course":
        search_query = f"{question}\n{course} curriculum modules topics syllabus"
    else:
        search_query = question

    context_docs = search_documents(
        search_query,
        k=settings.RETRIEVAL_K,
        course=course if course != "the course" else None,
        boost_terms=_detect_boost_terms(question),
    )

    answer = (
        default_fallback_message(course)
        if not context_docs
        else generate_answer(question, context_docs, course=course)
    )

    if session_id:
        session_store.update(
            session_id, question, answer, course=course if course != "the course" else None
        )

    return answer
