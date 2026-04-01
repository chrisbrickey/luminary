"""Tests for src/schemas/eval.py - evaluation harness schemas"""

from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.eval import GoldenDataset, GoldenExample, MetricResult

# --- Shared test constants ---

METRIC_NAME = "test_metric"
VALID_SCORE = 0.80
EXAMPLE_ID = "test_example_001"
QUESTION_TEXT = "What is the meaning of tolerance?"
DATASET_VERSION = "3.0"
DATASET_DATE = "2029-05-09"
DATASET_DESCRIPTION = "Test dataset for evaluation"


def _metric_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default MetricResult kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "name": METRIC_NAME,
        "score": VALID_SCORE,
    }
    defaults.update(overrides)
    return defaults


def _golden_example_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default GoldenExample kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "id": EXAMPLE_ID,
        "question": QUESTION_TEXT,
        "language": "en",
        "expected_chunk_ids": [],
    }
    defaults.update(overrides)
    return defaults


def _golden_dataset_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default GoldenDataset kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "version": DATASET_VERSION,
        "created_date": DATASET_DATE,
        "description": DATASET_DESCRIPTION,
        "examples": [],
    }
    defaults.update(overrides)
    return defaults


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


class TestGoldenExample:
    def test_construction_with_required_fields(self) -> None:
        example = GoldenExample(**_golden_example_kwargs())
        assert example.id == EXAMPLE_ID
        assert example.question == QUESTION_TEXT
        assert example.language == "en"
        assert example.expected_chunk_ids == []

    def test_construction_with_chunk_ids(self) -> None:
        chunk_ids = ["chunk001", "chunk002", "chunk003"]
        example = GoldenExample(**_golden_example_kwargs(expected_chunk_ids=chunk_ids))
        assert example.expected_chunk_ids == chunk_ids

    def test_language_french(self) -> None:
        example = GoldenExample(**_golden_example_kwargs(language="fr"))
        assert example.language == "fr"

    def test_language_invalid_pattern_raises(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            GoldenExample(**_golden_example_kwargs(language="ENG"))

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _golden_example_kwargs()
            del kwargs["id"]
            GoldenExample(**kwargs)  # type: ignore[call-arg]

    def test_missing_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _golden_example_kwargs()
            del kwargs["question"]
            GoldenExample(**kwargs)  # type: ignore[call-arg]

    def test_missing_language_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _golden_example_kwargs()
            del kwargs["language"]
            GoldenExample(**kwargs)  # type: ignore[call-arg]

    def test_expected_chunk_ids_defaults_to_empty_list(self) -> None:
        kwargs = _golden_example_kwargs()
        del kwargs["expected_chunk_ids"]
        example = GoldenExample(**kwargs)
        assert example.expected_chunk_ids == []


class TestGoldenDataset:
    def test_construction_with_required_fields(self) -> None:
        dataset = GoldenDataset(**_golden_dataset_kwargs())
        assert dataset.version == DATASET_VERSION
        assert dataset.created_date == DATASET_DATE
        assert dataset.description == DATASET_DESCRIPTION
        assert dataset.examples == []

    def test_construction_with_examples(self) -> None:
        example1 = GoldenExample(**_golden_example_kwargs(id="example_001"))
        example2 = GoldenExample(**_golden_example_kwargs(id="example_002", language="fr"))
        dataset = GoldenDataset(**_golden_dataset_kwargs(examples=[example1, example2]))
        assert len(dataset.examples) == 2
        assert dataset.examples[0].id == "example_001"
        assert dataset.examples[1].id == "example_002"

    def test_version_invalid_pattern_raises(self) -> None:
        with pytest.raises(ValidationError, match="pattern"):
            GoldenDataset(**_golden_dataset_kwargs(version="v1.0"))

    def test_version_valid_pattern_passes(self) -> None:
        dataset = GoldenDataset(**_golden_dataset_kwargs(version="2.0"))
        assert dataset.version == "2.0"

    def test_missing_version_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _golden_dataset_kwargs()
            del kwargs["version"]
            GoldenDataset(**kwargs)  # type: ignore[call-arg]

    def test_missing_created_date_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _golden_dataset_kwargs()
            del kwargs["created_date"]
            GoldenDataset(**kwargs)  # type: ignore[call-arg]

    def test_missing_description_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _golden_dataset_kwargs()
            del kwargs["description"]
            GoldenDataset(**kwargs)  # type: ignore[call-arg]
