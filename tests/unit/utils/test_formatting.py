"""Tests for formatting utilities."""

import pytest

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.schemas import ChatResponse
from src.utils.formatting import deduplicate_sources, format_sources


@pytest.fixture
def response_no_sources() -> ChatResponse:
    """ChatResponse with no sources."""
    return ChatResponse(
        text="Answer text",
        retrieved_contexts=[],
        retrieved_passage_ids=[],
        retrieved_source_titles=[],
        language=ENGLISH_ISO_CODE,
    )


@pytest.fixture
def response_with_sources() -> ChatResponse:
    """ChatResponse with multiple sources."""
    return ChatResponse(
        text="Answer text",
        retrieved_contexts=["Context A", "Context B", "Context C"],
        retrieved_passage_ids=["id1", "id2", "id3"],
        retrieved_source_titles=["Source A", "Source B", "Source C"],
        language=ENGLISH_ISO_CODE,
    )


@pytest.fixture
def response_with_duplicate_sources() -> ChatResponse:
    """ChatResponse with duplicate source titles."""
    return ChatResponse(
        text="Answer text",
        retrieved_contexts=["Context A", "Context B", "Context C", "Context D"],
        retrieved_passage_ids=["id1", "id2", "id3", "id4"],
        retrieved_source_titles=["Source A", "Source B", "Source A", "Source C"],
        language=ENGLISH_ISO_CODE,
    )


class TestDeduplicateSources:
    """Tests for deduplicate_sources function."""

    def test_empty_sources(self, response_no_sources: ChatResponse) -> None:
        """Test deduplication with no sources."""
        result = deduplicate_sources(response_no_sources)
        assert result == []

    def test_unique_sources(self, response_with_sources: ChatResponse) -> None:
        """Test deduplication with all unique sources."""
        result = deduplicate_sources(response_with_sources)
        assert result == ["Source A", "Source B", "Source C"]

    def test_duplicate_sources(
        self, response_with_duplicate_sources: ChatResponse
    ) -> None:
        """Test deduplication removes duplicates and preserves order."""
        result = deduplicate_sources(response_with_duplicate_sources)
        assert result == ["Source A", "Source B", "Source C"]

    def test_preserves_first_occurrence(self) -> None:
        """Test that first occurrence is preserved when deduplicating."""
        response = ChatResponse(
            text="Answer",
            retrieved_contexts=["A", "B", "C"],
            retrieved_passage_ids=["1", "2", "3"],
            retrieved_source_titles=["First", "Second", "First"],
            language=ENGLISH_ISO_CODE,
        )
        result = deduplicate_sources(response)
        # First occurrence of "First" should be at index 0
        assert result == ["First", "Second"]


class TestFormatSources:
    """Tests for format_sources function."""

    def test_no_sources_english(self, response_no_sources: ChatResponse) -> None:
        """Test format with no sources in English."""
        result = format_sources(response_no_sources, ENGLISH_ISO_CODE)
        assert result == "**Sources:** none"

    def test_no_sources_french(self, response_no_sources: ChatResponse) -> None:
        """Test format with no sources in French."""
        result = format_sources(response_no_sources, FRENCH_ISO_CODE)
        assert result == "**Sources :** aucune"

    def test_with_sources_english(self, response_with_sources: ChatResponse) -> None:
        """Test format with sources in English."""
        result = format_sources(response_with_sources, ENGLISH_ISO_CODE)
        expected = "**Sources:**\n- Source A\n- Source B\n- Source C"
        assert result == expected

    def test_with_sources_french(self, response_with_sources: ChatResponse) -> None:
        """Test format with sources in French."""
        result = format_sources(response_with_sources, FRENCH_ISO_CODE)
        expected = "**Sources :**\n- Source A\n- Source B\n- Source C"
        assert result == expected

    def test_deduplicates_sources(
        self, response_with_duplicate_sources: ChatResponse
    ) -> None:
        """Test that format_sources deduplicates before formatting."""
        result = format_sources(response_with_duplicate_sources, ENGLISH_ISO_CODE)
        # Should only show unique sources: A, B, C (not A twice)
        expected = "**Sources:**\n- Source A\n- Source B\n- Source C"
        assert result == expected

    def test_uses_default_language(self, response_with_sources: ChatResponse) -> None:
        """Test that default language parameter works."""
        # Not passing language should use DEFAULT_RESPONSE_LANGUAGE (en)
        result = format_sources(response_with_sources)
        expected = "**Sources:**\n- Source A\n- Source B\n- Source C"
        assert result == expected

    def test_with_single_source_english(self) -> None:
        """Test format with exactly one source in English."""
        response = ChatResponse(
            text="Answer",
            retrieved_contexts=["Context"],
            retrieved_passage_ids=["id1"],
            retrieved_source_titles=["Only Source"],
            language=ENGLISH_ISO_CODE,
        )
        result = format_sources(response, ENGLISH_ISO_CODE)
        expected = "**Sources:**\n- Only Source"
        assert result == expected

    def test_with_single_source_french(self) -> None:
        """Test format with exactly one source in French."""
        response = ChatResponse(
            text="Réponse",
            retrieved_contexts=["Contexte"],
            retrieved_passage_ids=["id1"],
            retrieved_source_titles=["Seule Source"],
            language=FRENCH_ISO_CODE,
        )
        result = format_sources(response, FRENCH_ISO_CODE)
        expected = "**Sources :**\n- Seule Source"
        assert result == expected
