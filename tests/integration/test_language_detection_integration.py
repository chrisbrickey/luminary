"""Integration tests for language detection through the full chain.

These tests verify that language detection flows correctly through the system
without making real LLM calls. They use FakeChatModel to avoid external dependencies
while testing the full chain integration.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.chains.chat_chain import build_chain
from src.configs.authors import DEFAULT_AUTHOR
from src.configs.common import (
    DEFAULT_RESPONSE_LANGUAGE,
    ENGLISH_ISO_CODE,
    FRENCH_ISO_CODE,
    ITALIAN_ISO_CODE,
    SPANISH_ISO_CODE,
)
from tests.conftest import FakeChatModel, FakeEmbeddings

# Test questions in different languages
FRENCH_QUESTION = "Quelle est votre opinion sur la tolérance religieuse et la liberté de conscience?"
ENGLISH_QUESTION = "What is your view on religious tolerance and freedom of conscience?"
ITALIAN_QUESTION = "Qual è la tua opinione sulla tolleranza religiosa e la libertà di coscienza?"

@pytest.fixture
def test_retriever(setup_test_db: tuple[Path, FakeEmbeddings], make_test_document):
    """Build retriever with sample document for language detection tests."""
    from src.vectorstores.chroma import embed_and_store
    from src.vectorstores.retriever import build_retriever

    db_path, embeddings = setup_test_db

    # Create sample documents
    docs = [
        make_test_document(
            content="Religious tolerance is essential for social harmony.",
            chunk_id="test_001",
            chunk_index=0,
            doc_id="test-doc",
            title="Test Document",
            author=DEFAULT_AUTHOR,
            page_number=1,
        ),
    ]

    # Embed and store
    embed_and_store(docs, embeddings=embeddings)

    return build_retriever(
        author=DEFAULT_AUTHOR,
        embeddings=embeddings,
    )

class TestLanguageDetectionIntegration:
    """Integration tests for language detection through the chain."""

    def test_french_question_detected_as_french(self, test_retriever) -> None:
        """Should detect French question and pass 'fr' to chain."""
        from src.utils.language import detect_language

        # Detect language from question
        detected_lang = detect_language(FRENCH_QUESTION)

        # Build and invoke chain with detected language
        chain = build_chain(retriever=test_retriever, llm=FakeChatModel())
        response = chain.invoke(FRENCH_QUESTION, language=detected_lang)

        # Verify language was detected and set in response
        assert response.language == FRENCH_ISO_CODE
        assert response.text is not None
        assert len(response.retrieved_passage_ids) > 0

    def test_english_question_detected_as_english(self, test_retriever) -> None:
        """Should detect English question and pass 'en' to chain."""
        from src.utils.language import detect_language

        # Detect language from question
        detected_lang = detect_language(ENGLISH_QUESTION)

        # Build and invoke chain with detected language
        chain = build_chain(retriever=test_retriever, llm=FakeChatModel())
        response = chain.invoke(ENGLISH_QUESTION, language=detected_lang)

        # Verify language was detected and set in response
        assert response.language == ENGLISH_ISO_CODE
        assert response.text is not None

    def test_italian_question_detected_as_italian(self, test_retriever) -> None:
        """Should detect Italian (not in LOCALIZATION_LANGUAGES) and still pass 'it' to chain.

        This tests that detection works for any language, even if UI strings
        aren't available for it. The UI will fall back to English for formatting,
        but the LLM should still respond in Italian.
        """
        from src.utils.language import detect_language

        # Detect language from question
        detected_lang = detect_language(ITALIAN_QUESTION)

        # Build and invoke chain with detected language
        chain = build_chain(retriever=test_retriever, llm=FakeChatModel())
        response = chain.invoke(ITALIAN_QUESTION, language=detected_lang)

        # Verify Italian was detected (even though not in LOCALIZATION_LANGUAGES)
        assert response.language == ITALIAN_ISO_CODE
        assert response.text is not None

    def test_config_overrides_detection(self, test_retriever) -> None:
        """Should use config language instead of detecting when config provided."""
        # Build chain
        chain = build_chain(retriever=test_retriever, llm=FakeChatModel())

        # English question but language parameter forces Spanish
        response = chain.invoke(ENGLISH_QUESTION, language=SPANISH_ISO_CODE)

        # Should use config, not detected language
        assert response.language == SPANISH_ISO_CODE

    @patch("src.utils.language.detect_langs")
    def test_detection_failure_fallback(
        self, mock_detect_langs, test_retriever
    ) -> None:
        """Should fall back to DEFAULT_RESPONSE_LANGUAGE when detection fails.

        This tests the safety fallback when langdetect library returns empty
        results or raises an exception.
        """
        from src.utils.language import detect_language

        # Simulate detection failure (empty results)
        mock_detect_langs.return_value = []

        # Detect language (which should fail and return default)
        detected_lang = detect_language("Some text that fails detection")

        # Build and invoke chain with detected (fallback) language
        chain = build_chain(retriever=test_retriever, llm=FakeChatModel())
        response = chain.invoke("Some text that fails detection", language=detected_lang)

        # Should fall back to default
        assert response.language == DEFAULT_RESPONSE_LANGUAGE
