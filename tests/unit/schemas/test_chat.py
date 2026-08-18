"""Unit tests for chat response schemas"""

from typing import Any

import pytest
from pydantic import ValidationError

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.schemas.chat import ChatResponse, SourceReference

# --- Shared test constants ---

CHUNK_ID = "abc123def456"
SAMPLE_TITLE = "Sample Work"


def _chat_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default ChatResponse kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "text": "Some text.",
        "retrieved_passage_ids": [],
        "retrieved_contexts": [],
        "retrieved_source_titles": [],
        "language": FRENCH_ISO_CODE,
    }
    defaults.update(overrides)
    return defaults


class TestChatResponse:
    def test_construction(self) -> None:
        response = ChatResponse(
            **_chat_kwargs(
                text="La tolérance est une vertu cardinale.",
                retrieved_passage_ids=[CHUNK_ID, "fedcba654321"],
                retrieved_contexts=["Context passage one.", "Context passage two."],
                retrieved_source_titles=["Lettres philosophiques, p. 1"],
            )
        )
        assert response.text == "La tolérance est une vertu cardinale."
        assert response.language == FRENCH_ISO_CODE
        assert len(response.retrieved_passage_ids) == 2

    def test_language_valid_english(self) -> None:
        response = ChatResponse(
            **_chat_kwargs(text="Tolerance is a cardinal virtue.", language=ENGLISH_ISO_CODE)
        )
        assert response.language == ENGLISH_ISO_CODE

    def test_language_must_be_two_lowercase_letters(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            ChatResponse(**_chat_kwargs(language="FR"))

    def test_language_too_long_raises(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            ChatResponse(**_chat_kwargs(language="fra"))

    def test_language_empty_raises(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            ChatResponse(**_chat_kwargs(language=""))

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _chat_kwargs()
            del kwargs["text"]
            ChatResponse(**kwargs)  # type: ignore[call-arg]

    def test_retrieved_lists_can_be_empty(self) -> None:
        response = ChatResponse(**_chat_kwargs(text="Réponse sans sources."))
        assert response.retrieved_passage_ids == []
        assert response.retrieved_contexts == []
        assert response.retrieved_source_titles == []

    def test_retrieved_sources_defaults_to_empty_list_for_legacy_kwargs(self) -> None:
        """_chat_kwargs omits retrieved_sources, so old eval artifacts must still deserialize."""
        response = ChatResponse(**_chat_kwargs())
        assert response.retrieved_sources == []

    def test_retrieved_sources_deserializes_from_dicts(self) -> None:
        """JSON-loaded artifacts pass dicts, which pydantic must validate into SourceReference."""
        response = ChatResponse(
            **_chat_kwargs(retrieved_sources=[{"title": "X", "page_number": 3}])
        )
        assert response.retrieved_sources == [SourceReference(title="X", page_number=3)]


class TestSourceReference:
    def test_default_page_number_is_none(self) -> None:
        source = SourceReference(title=SAMPLE_TITLE)
        assert source.page_number is None

    def test_int_page_number_is_preserved(self) -> None:
        source = SourceReference(title=SAMPLE_TITLE, page_number=12)
        assert source.page_number == 12

    def test_all_digit_string_coerces_to_int(self) -> None:
        source = SourceReference(title=SAMPLE_TITLE, page_number="12")
        assert source.page_number == 12

    def test_non_digit_string_coerces_to_none(self) -> None:
        source = SourceReference(title=SAMPLE_TITLE, page_number="xii")
        assert source.page_number is None

    def test_float_coerces_to_none(self) -> None:
        source = SourceReference(title=SAMPLE_TITLE, page_number=12.5)
        assert source.page_number is None

    def test_bool_coerces_to_none(self) -> None:
        source = SourceReference(title=SAMPLE_TITLE, page_number=True)
        assert source.page_number is None

    def test_page_number_zero_is_preserved_not_none(self) -> None:
        """Page 0 is falsy but valid; the coercion must not treat it as missing."""
        source = SourceReference(title=SAMPLE_TITLE, page_number=0)
        assert source.page_number == 0
