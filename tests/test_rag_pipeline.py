"""Unit tests for course detection and the ask_rag orchestration (LLM/DB mocked)."""
from unittest.mock import patch

from app.core.session_store import session_store
from app.retrieval.rag_pipeline import (
    ACK_RESPONSES,
    CLARIFICATION_RESPONSES,
    FAREWELL_RESPONSES,
    GREETING_RESPONSES,
    THANKS_RESPONSES,
    _detect_boost_terms,
    _looks_like_followup,
    ask_rag,
    extract_course_name,
)


class TestCourseDetection:
    def test_detects_data_science(self):
        assert extract_course_name("What is in the Data Science course?") == "Data Science"

    def test_detects_artificial_intelligence_full(self):
        assert extract_course_name("Tell me about Artificial Intelligence") == "Artificial Intelligence"

    def test_bare_ai_word(self):
        assert extract_course_name("what does the AI course teach") == "Artificial Intelligence"

    def test_ai_not_matched_inside_training(self):
        # "training" contains "ai" but must NOT trigger AI course detection.
        assert extract_course_name("what is the training schedule") == "the course"

    def test_detects_soc_analyst(self):
        assert extract_course_name("SOC Analyst syllabus") == "SOC Analyst"

    def test_no_course_returns_default(self):
        assert extract_course_name("hello there") == "the course"

    def test_unrelated_for_phrase_not_hallucinated(self):
        # "for/in/about <phrase>" alone must NOT be treated as a course name
        # unless followed by course/program/bootcamp/curriculum/syllabus.
        assert extract_course_name("What courses do you have for beginners?") == "the course"
        assert extract_course_name("Tell me about the weather today") == "the course"
        assert extract_course_name("What is the fee structure for working professionals?") == "the course"

    def test_fallback_still_detects_named_course_phrasing(self):
        assert extract_course_name("what is taught in the react course") == "React"


class TestBoostTermDetection:
    def test_location_intent(self):
        terms = _detect_boost_terms("what is the location?")
        assert "location" in terms and "branch" in terms

    def test_contact_intent(self):
        terms = _detect_boost_terms("how can I contact you")
        assert "contact" in terms

    def test_no_intent_returns_empty(self):
        assert _detect_boost_terms("what does the AI course teach") == []


class TestFollowupDetection:
    def test_short_question_is_followup(self):
        assert _looks_like_followup("what about fees?")

    def test_long_standalone_question_is_not_followup(self):
        assert not _looks_like_followup(
            "Can you give me a complete breakdown of everything covered in the SOC Analyst program"
        )


class TestAskRag:
    def test_empty_question(self):
        assert ask_rag("   ") == "Please ask a specific question."

    @patch("app.retrieval.rag_pipeline.search_documents", return_value=[])
    def test_no_context_found(self, _mock_search):
        result = ask_rag("Tell me about Data Science")
        assert "couldn't find" in result.lower()

    @patch("app.retrieval.rag_pipeline.generate_answer", return_value="Mocked answer")
    @patch("app.retrieval.rag_pipeline.search_documents", return_value=["ctx1", "ctx2"])
    def test_happy_path(self, mock_search, mock_llm):
        result = ask_rag("What is in Data Science?")
        assert result == "Mocked answer"
        mock_search.assert_called_once()
        mock_llm.assert_called_once()


class TestDialogueActShortCircuits:
    """Smalltalk (greetings, acknowledgments, thanks, farewells, bare yes/no)
    must never reach retrieval/generation -- there's nothing in a course
    catalog that grounds a reply to "ok", and running it through RAG anyway
    just burns a Groq call to produce the same canned refusal."""

    @patch("app.retrieval.rag_pipeline.search_documents")
    @patch("app.retrieval.rag_pipeline.generate_answer")
    def test_greetings_short_circuit(self, mock_llm, mock_search):
        for greeting in ["hi", "Hi!", "hello", "  HELLO  ", "hey", "good morning"]:
            assert ask_rag(greeting) in GREETING_RESPONSES
        mock_search.assert_not_called()
        mock_llm.assert_not_called()

    @patch("app.retrieval.rag_pipeline.search_documents")
    @patch("app.retrieval.rag_pipeline.generate_answer")
    def test_acknowledgments_short_circuit(self, mock_llm, mock_search):
        for word in ["ok", "Okay", "cool", "got it", "sure"]:
            assert ask_rag(word) in ACK_RESPONSES
        mock_search.assert_not_called()
        mock_llm.assert_not_called()

    @patch("app.retrieval.rag_pipeline.search_documents")
    @patch("app.retrieval.rag_pipeline.generate_answer")
    def test_thanks_short_circuit(self, mock_llm, mock_search):
        for word in ["thanks", "Thank you", "thanks a lot"]:
            assert ask_rag(word) in THANKS_RESPONSES
        mock_search.assert_not_called()
        mock_llm.assert_not_called()

    @patch("app.retrieval.rag_pipeline.search_documents")
    @patch("app.retrieval.rag_pipeline.generate_answer")
    def test_farewells_short_circuit(self, mock_llm, mock_search):
        for word in ["bye", "goodbye", "take care"]:
            assert ask_rag(word) in FAREWELL_RESPONSES
        mock_search.assert_not_called()
        mock_llm.assert_not_called()

    @patch("app.retrieval.rag_pipeline.search_documents")
    @patch("app.retrieval.rag_pipeline.generate_answer")
    def test_bare_yes_no_asks_for_clarification(self, mock_llm, mock_search):
        for word in ["yes", "no", "maybe"]:
            assert ask_rag(word) in CLARIFICATION_RESPONSES
        mock_search.assert_not_called()
        mock_llm.assert_not_called()

    @patch("app.retrieval.rag_pipeline.generate_answer", return_value="Mocked answer")
    @patch("app.retrieval.rag_pipeline.search_documents", return_value=["ctx1"])
    def test_smalltalk_word_combined_with_real_question_is_not_short_circuited(self, mock_search, mock_llm):
        # Only a *pure* smalltalk message should short-circuit; smalltalk
        # attached to an actual question must still go through full RAG.
        result = ask_rag("hi, what does the AI course cover?")
        assert result == "Mocked answer"
        mock_search.assert_called_once()


class TestConversationMemory:
    @patch("app.retrieval.rag_pipeline.generate_answer", return_value="Mocked DS answer")
    @patch("app.retrieval.rag_pipeline.search_documents", return_value=["ctx1"])
    def test_session_remembers_last_course_for_followup(self, mock_search, _mock_llm):
        session_id = "test-session-followup"
        session_store._sessions.pop(session_id, None)

        ask_rag("What is in the Data Science course?", session_id=session_id)
        # Follow-up has no course of its own -- should reuse Data Science
        # from the session instead of searching with no course context.
        ask_rag("what about the fees?", session_id=session_id)

        second_call_kwargs = mock_search.call_args_list[1].kwargs
        assert second_call_kwargs["course"] == "Data Science"

    @patch("app.retrieval.rag_pipeline.generate_answer", return_value="Mocked answer")
    @patch("app.retrieval.rag_pipeline.search_documents", return_value=["ctx1"])
    def test_no_session_id_means_no_memory(self, mock_search, _mock_llm):
        ask_rag("What is in the Data Science course?")
        ask_rag("what about the fees?")

        second_call_kwargs = mock_search.call_args_list[1].kwargs
        assert second_call_kwargs["course"] is None
