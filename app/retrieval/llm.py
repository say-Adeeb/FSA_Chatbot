"""LLM answer generation via Groq, grounded strictly in retrieved context."""
import logging

from groq import Groq

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def default_fallback_message(course: str = "the course") -> str:
    if course != "the course":
        return f"I couldn't find detailed curriculum information for {course}."
    return (
        "I couldn't find information about that. Try asking about a specific course "
        "(Data Science, Data Analyst, SOC Analyst, Artificial Intelligence, and more), "
        "or about our locations, contact details, or admissions."
    )


def generate_answer(question: str, context_docs: list[str], course: str = "the course") -> str:
    context = "\n\n---\n\n".join(doc.strip() for doc in context_docs if doc and doc.strip())
    fallback = default_fallback_message(course)
    is_specific_course = course != "the course"

    if is_specific_course:
        system_message = (
            "You are a highly precise EdTech assistant for Full Stack Academy. "
            "Answer ONLY using the provided context. "
            f"If the context lists specific modules, topics, or curriculum for {course}, summarize ONLY those. "
            f"If the context does not contain the answer, reply exactly: '{fallback}' "
            f"Never invent, generalize, or add topics that are not explicitly present in the context for {course}. "
            "The context is scraped from a marketing website and may mix real curriculum bullet points with "
            "promotional noise (student/placement counts, ratings, 'Book Now' calls-to-action, batch timings) "
            "and mentions of OTHER courses. Ignore all of that noise: only report items that are genuinely "
            f"presented as a topic, module, or curriculum item for {course} specifically. If a tool, technology, "
            f"or topic is only associated with a different named course in the context, do not attribute it to {course}."
        )
    else:
        # No specific course was detected -- this may be a general question about
        # locations, contact info, admissions, fees, etc. Framing the prompt around
        # "curriculum for a course" here would make the model refuse to use context
        # that plainly answers the question just because it isn't a course module.
        system_message = (
            "You are a precise assistant for Full Stack Academy, an ed-tech training institute. "
            "Answer ONLY using the provided context, which may cover courses offered, branch "
            "locations, contact details, admissions, fees, or other general academy information. "
            f"If the context does not contain the answer, reply exactly: '{fallback}' "
            "Never invent facts that are not explicitly present in the context. "
            "The context is scraped from a marketing website and may mix genuine factual content "
            "with promotional noise (ratings, 'Book Now' calls-to-action, batch timings) -- ignore "
            "the noise and answer using only the genuine factual content."
        )

    user_prompt = (
        "Answer the user's question using only the context below.\n"
        f"If the answer is not in the context, reply exactly: {fallback}\n"
        "Do not add unrelated or invented information.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question:\n{question}"
    )

    try:
        response = _get_client().chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.LLM_TEMPERATURE,
            top_p=0.95,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    except Exception:
        logger.exception("Groq request failed")
        return "Sorry, I'm having trouble answering right now. Please try again shortly."

    content = None
    if response.choices:
        message = getattr(response.choices[0], "message", None)
        content = getattr(message, "content", None)

    return content.strip() if content else fallback
