"""LLM-as-judge: scores a generated answer for groundedness and relevance."""
import json
import logging
import re

from groq import Groq

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Groq | None = None

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for a RAG chatbot. You will be shown the retrieved "
    "context, the user's question, and the chatbot's answer. Judge two things:\n"
    "1. grounded: true only if every claim in the answer is directly supported by the "
    "context (no invented facts, no outside knowledge). If the answer correctly says "
    "it couldn't find the information rather than guessing, that counts as grounded=true.\n"
    "2. relevant: true if the answer actually addresses the user's question (a correct "
    "refusal to an off-topic or unanswerable question also counts as relevant=true).\n"
    "Respond with ONLY a JSON object, no markdown fences: "
    '{"grounded": true/false, "relevant": true/false, "reasoning": "<one short sentence>"}'
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _get_judge_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def judge_answer(question: str, context: str, answer: str) -> dict:
    prompt = (
        f"Context:\n{context or '(no context retrieved)'}\n\n"
        f"Question:\n{question}\n\n"
        f"Chatbot Answer:\n{answer}"
    )

    try:
        response = _get_judge_client().chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        match = _JSON_BLOCK.search(content)
        if not match:
            raise ValueError(f"No JSON object found in judge response: {content!r}")
        parsed = json.loads(match.group(0))
        return {
            "grounded": bool(parsed["grounded"]),
            "relevant": bool(parsed["relevant"]),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception:
        logger.exception("Judge call failed for question: %s", question)
        return {"grounded": None, "relevant": None, "reasoning": "judge_error"}
