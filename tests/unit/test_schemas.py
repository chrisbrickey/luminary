"""Tests for src/schemas.py - validated data structures used in the application"""

from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas import ChatResponse, ChunkInfo, MetricResult

# -- Shared test constants --------------------------------------------------

CHUNK_ID = "abc123def456"
DOCUMENT_ID = "some-document"
DOCUMENT_TITLE = "The Title of Some Document"
AUTHOR = "some author"
SOURCE_URL = "https://somesource.org/documents"
METRIC_NAME = "test_metric"
VALID_SCORE = 0.75


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


def _metric_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default MetricResult kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "name": METRIC_NAME,
        "score": VALID_SCORE,
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


class TestMetricResult:
    def test_construction_with_required_fields(self) -> None:
        metric = MetricResult(**_metric_kwargs())
        assert metric.name == METRIC_NAME
        assert metric.score == VALID_SCORE
        assert metric.details == {}

    def test_construction_with_details(self) -> None:
        test_details = {"reason": "sample reason", "count": 42}
        metric = MetricResult(**_metric_kwargs(details=test_details))
        assert metric.details == test_details

    def test_score_at_lower_bound(self) -> None:
        metric = MetricResult(**_metric_kwargs(score=0.0))
        assert metric.score == 0.0

    def test_score_at_upper_bound(self) -> None:
        metric = MetricResult(**_metric_kwargs(score=1.0))
        assert metric.score == 1.0

    def test_score_below_lower_bound_raises(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            MetricResult(**_metric_kwargs(score=-0.1))

    def test_score_above_upper_bound_raises(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            MetricResult(**_metric_kwargs(score=1.1))

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricResult(score=VALID_SCORE)

    def test_missing_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            MetricResult(name=METRIC_NAME)

    def test_score_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            MetricResult(name=METRIC_NAME, score="not-a-float")

    def test_details_defaults_to_empty_dict(self) -> None:
        metric = MetricResult(name="sample_metric", score=0.5)
        assert metric.details == {}
