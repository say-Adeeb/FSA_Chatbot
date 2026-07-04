"""Unit tests for the fallback-message logic in app.retrieval.llm."""
from app.retrieval.llm import default_fallback_message


def test_course_specific_fallback_names_the_course():
    msg = default_fallback_message("Data Science")
    assert "Data Science" in msg
    assert "couldn't find" in msg.lower()


def test_generic_fallback_does_not_claim_curriculum_scope():
    # Regression: when no course is detected, the message used to say
    # "I couldn't find detailed curriculum information for the course",
    # which is misleading for non-curriculum questions (location, fees,
    # contact info) that the bot might genuinely be unable to answer.
    msg = default_fallback_message("the course")
    assert "curriculum" not in msg.lower()
    assert "couldn't find" in msg.lower()
