"""Tests for src/schemas/chat.py - chat response schemas"""

from typing import Any

import pytest
from pydantic import ValidationError

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.schemas.chat import ChatResponse

# --- Shared test constants ---

CHUNK_ID = "abc123def456"


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
