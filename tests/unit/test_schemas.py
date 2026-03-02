"""Tests for src/schemas.py — ChunkInfo and ChatResponse."""

from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas import ChatResponse, ChunkInfo

# -- Shared test constants --------------------------------------------------

CHUNK_ID = "abc123def456"
DOCUMENT_ID = "some-document"
DOCUMENT_TITLE = "The Title of Some Document"
AUTHOR = "some author"
SOURCE_URL = "https://somesource.org/documents"


def _chunk_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default ChunkInfo kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "chunk_id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "chunk_index": 0,
        "source": SOURCE_URL,
    }
    defaults.update(overrides)
    return defaults


def _chat_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default ChatResponse kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "text": "Some text.",
        "retrieved_passage_ids": [],
        "retrieved_contexts": [],
        "retrieved_source_titles": [],
        "language": "fr",
    }
    defaults.update(overrides)
    return defaults


class TestChunkInfo:
    def test_construction_with_required_fields(self) -> None:
        chunk = ChunkInfo(**_chunk_kwargs(source=SOURCE_URL))
        assert chunk.chunk_id == CHUNK_ID
        assert chunk.document_id == DOCUMENT_ID
        assert chunk.document_title is None
        assert chunk.author is None
        assert chunk.chunk_index == 0
        assert chunk.source == SOURCE_URL

    def test_optional_fields_set(self) -> None:
        chunk = ChunkInfo(
            **_chunk_kwargs(
                document_title=DOCUMENT_TITLE,
                author=AUTHOR,
                source=SOURCE_URL,
            )
        )
        assert chunk.document_title == DOCUMENT_TITLE
        assert chunk.author == AUTHOR

    def test_author_must_be_lowercase(self) -> None:
        with pytest.raises(ValidationError, match="author must be lowercase"):
            ChunkInfo(**_chunk_kwargs(author="Voltaire"))

    def test_author_none_passes_validation(self) -> None:
        chunk = ChunkInfo(**_chunk_kwargs())
        assert chunk.author is None

    def test_extra_fields_allowed(self) -> None:
        chunk = ChunkInfo(**_chunk_kwargs(custom_field="extra_value"))
        assert chunk.model_extra is not None
        assert chunk.model_extra["custom_field"] == "extra_value"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _chunk_kwargs(
                document_title=DOCUMENT_TITLE,
                author=AUTHOR,
            )
            del kwargs["document_id"]
            ChunkInfo(**kwargs)  # type: ignore[arg-type]

    def test_chunk_index_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            ChunkInfo(**_chunk_kwargs(chunk_index="not-an-int"))  # type: ignore[arg-type]


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
        assert response.language == "fr"
        assert len(response.retrieved_passage_ids) == 2

    def test_language_valid_english(self) -> None:
        response = ChatResponse(
            **_chat_kwargs(text="Tolerance is a cardinal virtue.", language="en")
        )
        assert response.language == "en"

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
