"""External tests for language detection with real LLM.

These tests verify that the real Ollama LLM respects the language
parameter in prompts and responds in the detected language.
Local LLMs are not deterministic, especially with non-major languages.
To avoid flakiness, these tests do not use rare languages.

REQUIREMENTS:
- Ollama must be running: `ollama serve`
- Model must be pulled: `ollama pull mistral`
- ChromaDB must have ingested data: `uv run python scripts/ingest.py --author voltaire`

Run these tests explicitly:
    uv run pytest -m external tests/external/test_language_detection_external.py -v

These tests are EXCLUDED from the default test run.
"""

import pytest

from src.chains.chat_chain import build_chain, ChatResponse
from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE, GERMAN_ISO_CODE

# Mark all tests in this module as external
pytestmark = pytest.mark.external

# Test questions in different languages
FRENCH_QUESTION = "Selon vous,  uelle est la question la plus importante en philosophie ?"
ENGLISH_QUESTION = "In your view, what is the most important question in philosophy?"
GERMAN_QUESTION = "Was ist Ihrer Meinung nach die wichtigste Frage in der Philosophie?"


def assert_valid_response(response: ChatResponse, expected_language: str) -> None:
    """Assert basic response properties are valid."""
    assert response.language == expected_language
    assert response.text is not None
    assert len(response.text) > 0


def assert_response_contains_language_indicators(
    response_text: str,
    indicators: list[str],
    min_count: int,
    language_name: str,
) -> None:
    """Assert that response contains minimum number of language indicators.

    Args:
        response_text: The full response text to check
        indicators: List of language-specific indicators (e.g., [" the ", " is "])
        min_count: Minimum number of indicators that must be found
        language_name: Name of language for error messages
    """
    response_lower = response_text.lower()
    indicator_count = sum(
        1 for indicator in indicators if indicator in response_lower
    )
    assert (
        indicator_count >= min_count
    ), f"Response should contain {language_name} words, got: {response_text[:200]}..."


def assert_not_language(
    response_text: str, indicators: list[str], max_count: int, language_name: str
) -> None:
    """Assert that response does NOT appear to be in a different language.

    Args:
        response_text: Sample of response text to check
        indicators: List of language-specific indicators to check for
        max_count: Maximum count before flagging as false positive
        language_name: Name of language for error messages
    """
    response_lower = response_text.lower()
    indicator_count = sum(1 for indicator in indicators if indicator in response_lower)
    assert (
        indicator_count <= max_count
    ), f"Response appears to be in {language_name}, not the expected language"

class TestRealLLMLanguageHandling:
    """Tests with real Ollama LLM to verify language handling."""

    def test_french_question_llm_responds_in_french(self) -> None:
        """Should detect French question and LLM should respond in French.

        This test verifies the full end-to-end flow:
        1. User asks in French
        2. System detects 'fr'
        3. Chain passes 'fr' to LLM via prompt
        4. LLM actually responds in French
        """
        # Build and invoke chain
        chain = build_chain()
        response = chain.invoke(FRENCH_QUESTION, language=FRENCH_ISO_CODE)

        # Verify basic response properties
        assert_valid_response(response, FRENCH_ISO_CODE)

        # Verify response is actually in French (simple heuristic check)
        # French text typically contains common words like: la, le, de, les, est, que
        french_indicators = ["la ", "le ", " de ", " les ", " est ", " que ", " une "]
        assert_response_contains_language_indicators(
            response.text, french_indicators, min_count=2, language_name="French"
        )

    def test_english_question_llm_responds_in_english(self) -> None:
        """Should detect English question and LLM should respond in English."""
        # Build and invoke chain
        chain = build_chain()
        response = chain.invoke(ENGLISH_QUESTION, language=ENGLISH_ISO_CODE)

        # Verify basic response properties
        assert_valid_response(response, ENGLISH_ISO_CODE)

        # Verify response is actually in English (simple heuristic check)
        # English text typically contains common words: the, is, of, and, to, in, a
        english_indicators = [" the ", " is ", " of ", " and ", " to ", " in ", " a "]
        assert_response_contains_language_indicators(
            response.text, english_indicators, min_count=3, language_name="English"
        )

    def test_german_question_llm_responds_in_german(self) -> None:
        """Should detect German (unsupported for localization) and LLM should respond in German.

        This tests that even though German is not in LOCALIZATION_LANGUAGES
        (no localization of UI strings), the content retrieved from the LLM
        is in the detected language. The UI strings will fall back to the
        default language, but the LLM response content should be in German.
        """
        chain = build_chain()
        response = chain.invoke(GERMAN_QUESTION, language=GERMAN_ISO_CODE)

        # Verify basic response properties
        assert_valid_response(response, GERMAN_ISO_CODE)

        # Verify response is actually in German (simple heuristic check)
        german_indicators = [" der ", " die ", " das ", " ist ", " eine ", " und ", " den "]
        assert_response_contains_language_indicators(
            response.text, german_indicators, min_count=2, language_name="German"
        )

        # Additionally, verify it's not French or English
        # (could be a false positive if LLM ignores language instruction)
        response_sample = response.text[:300]
        assert_not_language(response_sample, [" est "], max_count=3, language_name="French")
        assert_not_language(response_sample, [" the "], max_count=3, language_name="English")
