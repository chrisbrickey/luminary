"""Tests for src/schemas/eval.py - evaluation harness schemas"""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.eval.metrics.base import FALLBACK_THRESHOLD
from src.schemas.chat import ChatResponse
from src.schemas.eval import EvalRun, ExampleResult, GoldenDataset, GoldenExample, MetricResult

# --- Shared test constants ---

METRIC_NAME = "test_metric"
VALID_SCORE = FALLBACK_THRESHOLD
EXAMPLE_ID = "test_example_001"
AUTHOR = "voltaire"
QUESTION_TEXT = "What is the meaning of tolerance?"
RUN_TIMESTAMP = "2029-05-09T14:30:45+00:00"

# Golden datasets
DATASET_SCOPE = "persona"
DATASET_VERSION = "3.0"
DATASET_DATE = "2029-05-09"
DATASET_DESCRIPTION = "Test dataset for evaluation"
DATASET_IDENTIFIER = f"{DATASET_SCOPE}_{AUTHOR}_v{DATASET_VERSION}_{DATASET_DATE}"


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
        "author": AUTHOR,
        "language": ENGLISH_ISO_CODE,
        "expected_chunk_ids": [],
    }
    defaults.update(overrides)
    return defaults


def _golden_dataset_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default GoldenDataset kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "scope": DATASET_SCOPE,
        "authors": [],
        "version": DATASET_VERSION,
        "created_date": DATASET_DATE,
        "description": DATASET_DESCRIPTION,
        "examples": [],
    }
    defaults.update(overrides)
    return defaults


def _chat_response_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default ChatResponse kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "text": "Sample response text.",
        "retrieved_passage_ids": ["chunk_001", "chunk_002"],
        "retrieved_contexts": ["Context 1", "Context 2"],
        "retrieved_source_titles": ["Source A", "Source B"],
        "language": ENGLISH_ISO_CODE,
    }
    defaults.update(overrides)
    return defaults


def _example_result_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default ExampleResult kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "example_id": EXAMPLE_ID,
        "question": QUESTION_TEXT,
        "language": ENGLISH_ISO_CODE,
        "response": ChatResponse(**_chat_response_kwargs()),
        "metrics": [],
        "passed": True,
    }
    defaults.update(overrides)
    return defaults


def _eval_run_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default EvalRun kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "dataset_scope": DATASET_SCOPE,
        "dataset_authors": [AUTHOR],
        "dataset_identifier": DATASET_IDENTIFIER,
        "dataset_version": DATASET_VERSION,
        "dataset_date": DATASET_DATE,
        "run_timestamp": RUN_TIMESTAMP,
        "system_version": {"chat_model": "test-model", "commit": "abc123"},
        "effective_thresholds": {METRIC_NAME: FALLBACK_THRESHOLD},
        "example_results": [],
        "aggregate_scores": {"overall": {}},
        "overall_pass_rate": 0.0,
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
        assert example.author == AUTHOR
        assert example.language == ENGLISH_ISO_CODE
        assert example.expected_chunk_ids == []

    def test_construction_with_chunk_ids(self) -> None:
        chunk_ids = ["chunk001", "chunk002", "chunk003"]
        example = GoldenExample(**_golden_example_kwargs(expected_chunk_ids=chunk_ids))
        assert example.expected_chunk_ids == chunk_ids

    def test_language_french(self) -> None:
        example = GoldenExample(**_golden_example_kwargs(language=FRENCH_ISO_CODE))
        assert example.language == FRENCH_ISO_CODE

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

    def test_missing_author_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _golden_example_kwargs()
            del kwargs["author"]
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
        assert dataset.authors == []
        assert dataset.description == DATASET_DESCRIPTION
        assert dataset.examples == []

    def test_construction_with_examples(self) -> None:
        example1 = GoldenExample(**_golden_example_kwargs(id="example_001"))
        example2 = GoldenExample(**_golden_example_kwargs(id="example_002", language=FRENCH_ISO_CODE))
        # Both examples have author=AUTHOR, so authors field should be [AUTHOR]
        dataset = GoldenDataset(**_golden_dataset_kwargs(authors=[AUTHOR], examples=[example1, example2]))
        assert len(dataset.examples) == 2
        assert dataset.examples[0].id == "example_001"
        assert dataset.examples[1].id == "example_002"
        assert dataset.authors == [AUTHOR]

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

    def test_missing_authors_raises(self) -> None:
        """Missing authors field raises ValidationError."""
        with pytest.raises(ValidationError):
            kwargs = _golden_dataset_kwargs()
            del kwargs["authors"]
            GoldenDataset(**kwargs)  # type: ignore[call-arg]

    def test_invalid_author_in_example_raises(self) -> None:
        """GoldenExample with invalid author raises ValidationError."""
        with pytest.raises(ValidationError, match="not found in AUTHOR_CONFIGS"):
            GoldenExample(**_golden_example_kwargs(author="nonexistent_author"))

    def test_invalid_author_in_dataset_metadata_raises(self) -> None:
        """GoldenDataset with invalid author in metadata raises ValidationError."""
        with pytest.raises(ValidationError, match="not found in AUTHOR_CONFIGS"):
            GoldenDataset(**_golden_dataset_kwargs(authors=["invalid_author"]))

    def test_authors_mismatch_with_examples_raises(self) -> None:
        """GoldenDataset with mismatched authors field raises ValidationError."""
        example1 = GoldenExample(**_golden_example_kwargs(id="ex1", author=AUTHOR))
        # Metadata claims no authors, but example has voltaire
        with pytest.raises(ValidationError, match="Mismatch between dataset.authors and example authors"):
            GoldenDataset(**_golden_dataset_kwargs(authors=[], examples=[example1]))

    @pytest.mark.skip(reason="Requires multiple authors in AUTHOR_CONFIGS to test properly")
    def test_extra_author_in_metadata_raises(self) -> None:
        """GoldenDataset with extra author in metadata (not in examples) raises ValidationError."""
        # This test would work if we had multiple valid authors in AUTHOR_CONFIGS

    def test_authors_match_examples_passes(self) -> None:
        """GoldenDataset with matching authors field passes validation."""
        example1 = GoldenExample(**_golden_example_kwargs(id="ex1", author=AUTHOR))
        example2 = GoldenExample(**_golden_example_kwargs(id="ex2", author=AUTHOR))
        dataset = GoldenDataset(**_golden_dataset_kwargs(authors=[AUTHOR], examples=[example1, example2]))
        assert dataset.authors == [AUTHOR]
        assert len(dataset.examples) == 2

    def test_identifier_single_author(self) -> None:
        """GoldenDataset.identifier returns correct format for single author."""
        example = GoldenExample(**_golden_example_kwargs(author=AUTHOR))
        dataset = GoldenDataset(**_golden_dataset_kwargs(authors=[AUTHOR], examples=[example]))
        expected_identifier = f"{DATASET_SCOPE}_{AUTHOR}_v{DATASET_VERSION}_{DATASET_DATE}"
        assert dataset.identifier == expected_identifier

    @patch("src.configs.authors.AUTHOR_CONFIGS", {"voltaire": Mock(), "gouges": Mock()})
    def test_authors_unsorted_raises_validation_error(self) -> None:
        """GoldenDataset.authors must be sorted alphabetically."""
        # Create examples for two authors
        example1 = GoldenExample(**_golden_example_kwargs(id="ex1", author="voltaire"))
        example2 = GoldenExample(**_golden_example_kwargs(id="ex2", author="gouges"))

        # Provide authors in reverse alphabetical order (unsorted)
        authors_unsorted = ["voltaire", "gouges"]

        # Assert: Unsorted authors raise ValidationError
        with pytest.raises(ValidationError, match="must be sorted alphabetically"):
            GoldenDataset(**_golden_dataset_kwargs(
                authors=authors_unsorted,
                examples=[example1, example2]
            ))

    @patch("src.configs.authors.AUTHOR_CONFIGS", {"voltaire": Mock(), "gouges": Mock()})
    def test_identifier_with_sorted_authors(self) -> None:
        """GoldenDataset.identifier correctly formats with sorted authors."""
        # Create examples for two authors
        example1 = GoldenExample(**_golden_example_kwargs(id="ex1", author="voltaire"))
        example2 = GoldenExample(**_golden_example_kwargs(id="ex2", author="gouges"))

        # Provide authors in sorted alphabetical order
        authors_sorted = ["gouges", "voltaire"]
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            authors=authors_sorted,
            examples=[example1, example2]
        ))

        # Assert: .authors field maintains sorted order
        assert dataset.authors == ["gouges", "voltaire"]

        # Assert: .identifier uses sorted authors (gouges < voltaire)
        expected_identifier = f"{DATASET_SCOPE}_gouges_voltaire_v{DATASET_VERSION}_{DATASET_DATE}"
        assert dataset.identifier == expected_identifier
        assert "gouges_voltaire" in dataset.identifier  # gouges comes first

    @patch("src.configs.authors.AUTHOR_CONFIGS", {"gouges": Mock(), "voltaire": Mock()})
    def test_identifier_sorts_authors_already_sorted_input(self) -> None:
        """GoldenDataset.identifier works correctly when authors already sorted."""
        # Create examples for two authors
        example1 = GoldenExample(**_golden_example_kwargs(id="ex1", author="gouges"))
        example2 = GoldenExample(**_golden_example_kwargs(id="ex2", author="voltaire"))

        # Provide authors already in alphabetical order
        authors_sorted = ["gouges", "voltaire"]
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            authors=authors_sorted,
            examples=[example1, example2]
        ))

        # Assert: .authors field preserves (already sorted) input order
        assert dataset.authors == ["gouges", "voltaire"]

        # Assert: .identifier maintains sorted order
        expected_identifier = f"{DATASET_SCOPE}_gouges_voltaire_v{DATASET_VERSION}_{DATASET_DATE}"
        assert dataset.identifier == expected_identifier

    @patch("src.configs.authors.AUTHOR_CONFIGS", {"author_a": Mock(), "author_b": Mock(), "author_c": Mock()})
    def test_identifier_with_three_sorted_authors(self) -> None:
        """GoldenDataset.identifier correctly formats with three sorted authors."""
        scope = "retrieval"
        version = "2.5"
        date = "2030-12-31"

        # Create examples for each author
        example_a = GoldenExample(**_golden_example_kwargs(id="ex_a", author="author_a"))
        example_b = GoldenExample(**_golden_example_kwargs(id="ex_b", author="author_b"))
        example_c = GoldenExample(**_golden_example_kwargs(id="ex_c", author="author_c"))

        # Provide authors in sorted alphabetical order
        authors_sorted = ["author_a", "author_b", "author_c"]
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            scope=scope,
            authors=authors_sorted,
            version=version,
            created_date=date,
            examples=[example_a, example_b, example_c]
        ))

        # Assert: .authors field maintains sorted order
        assert dataset.authors == ["author_a", "author_b", "author_c"]
        assert dataset.authors == sorted(dataset.authors)

        # Assert: .identifier uses sorted authors
        expected = f"{scope}_author_a_author_b_author_c_v{version}_{date}"
        assert dataset.identifier == expected

    @patch("src.configs.authors.AUTHOR_CONFIGS", {"author_a": Mock(), "author_b": Mock(), "author_c": Mock()})
    def test_identifier_format_structure(self) -> None:
        """GoldenDataset.identifier follows exact format: {scope}_{sorted_authors}_v{version}_{date}."""
        scope = "retrieval"
        authors = ["author_a", "author_b", "author_c"]
        version = "2.5"
        date = "2030-12-31"

        # Create examples for each author
        example_a = GoldenExample(**_golden_example_kwargs(id="ex_a", author="author_a"))
        example_b = GoldenExample(**_golden_example_kwargs(id="ex_b", author="author_b"))
        example_c = GoldenExample(**_golden_example_kwargs(id="ex_c", author="author_c"))

        dataset = GoldenDataset(**_golden_dataset_kwargs(
            scope=scope,
            authors=authors,
            version=version,
            created_date=date,
            examples=[example_a, example_b, example_c]
        ))

        # Verify exact format with sorted authors
        expected = f"{scope}_author_a_author_b_author_c_v{version}_{date}"
        assert dataset.identifier == expected

        # Verify structure components
        assert dataset.identifier.startswith(f"{scope}_")
        assert f"_v{version}_" in dataset.identifier
        assert dataset.identifier.endswith(f"_{date}")


class TestExampleResult:
    def test_construction_with_required_fields(self) -> None:
        """Valid fields construct successfully."""
        result = ExampleResult(**_example_result_kwargs())
        assert result.example_id == EXAMPLE_ID
        assert result.question == QUESTION_TEXT
        assert result.language == ENGLISH_ISO_CODE
        assert isinstance(result.response, ChatResponse)
        assert result.metrics == []
        assert result.passed is True

    def test_construction_with_all_fields(self) -> None:
        """All required fields are present."""
        metric1 = MetricResult(**_metric_kwargs(name="metric_a", score=0.9))
        metric2 = MetricResult(**_metric_kwargs(name="metric_b", score=0.7))
        result = ExampleResult(**_example_result_kwargs(
            metrics=[metric1, metric2],
            passed=False
        ))
        assert len(result.metrics) == 2
        assert result.metrics[0].name == "metric_a"
        assert result.metrics[1].name == "metric_b"
        assert result.passed is False

    def test_passed_field_calculated_correctly(self) -> None:
        """Passed field can be True or False."""
        result_passed = ExampleResult(**_example_result_kwargs(passed=True))
        assert result_passed.passed is True

        result_failed = ExampleResult(**_example_result_kwargs(passed=False))
        assert result_failed.passed is False


class TestEvalRun:
    def test_construction_with_required_fields(self) -> None:
        """Valid fields construct successfully."""
        run = EvalRun(**_eval_run_kwargs())

        # Dataset identification
        assert run.dataset_scope == DATASET_SCOPE
        assert run.dataset_authors == [AUTHOR]
        assert run.dataset_identifier == DATASET_IDENTIFIER
        assert run.dataset_version == DATASET_VERSION
        assert run.dataset_date == DATASET_DATE

        # Run metadata
        assert run.run_timestamp == RUN_TIMESTAMP
        assert run.system_version == {"chat_model": "test-model", "commit": "abc123"}
        assert run.effective_thresholds == {METRIC_NAME: FALLBACK_THRESHOLD}

        # Results
        assert run.example_results == []
        assert run.aggregate_scores == {"overall": {}}
        assert run.overall_pass_rate == 0.0

    def test_missing_dataset_version_raises(self) -> None:
        """Missing dataset_version raises error."""
        with pytest.raises(ValidationError):
            kwargs = _eval_run_kwargs()
            del kwargs["dataset_version"]
            EvalRun(**kwargs)  # type: ignore[call-arg]

    def test_aggregate_scores_structure(self) -> None:
        """Aggregate scores has expected nested structure."""
        aggregate_scores = {
            "overall": {"metric_a": 0.85, "metric_b": 0.90},
            "by_language": {
                ENGLISH_ISO_CODE: {"metric_a": 0.87, "metric_b": 0.92},
                FRENCH_ISO_CODE: {"metric_a": 0.83, "metric_b": 0.88}
            },
            "cross_language": {"translation_consistency": 0.75}
        }
        run = EvalRun(**_eval_run_kwargs(aggregate_scores=aggregate_scores))
        assert "overall" in run.aggregate_scores
        assert "by_language" in run.aggregate_scores
        assert "cross_language" in run.aggregate_scores
        assert run.aggregate_scores["by_language"][ENGLISH_ISO_CODE]["metric_a"] == 0.87
