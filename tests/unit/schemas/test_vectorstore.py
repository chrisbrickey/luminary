"""Tests for src/schemas/vectorstore.py - vector store and ingestion schemas"""

from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.vectorstore import ChunkInfo

# --- Shared test constants ---

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
