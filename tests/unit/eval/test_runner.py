"""Tests for src/eval/runner.py — evaluation runner."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.runnables import Runnable

from src.eval.metrics.base import MetricSpec
from src.schemas.chat import ChatResponse
from src.schemas.eval import EvalRun, GoldenDataset, GoldenExample, MetricResult


# -- Shared test constants and methods ---

EXAMPLE_ID_001 = "example_001"
EXAMPLE_ID_002 = "example_002"
EXAMPLE_ID_003 = "example_003"
QUESTION_001 = "What is tolerance?"
QUESTION_002 = "Qu'est-ce que la tolérance?"
QUESTION_003 = "What is reason?"
CHUNK_001 = "chunk_001"
CHUNK_002 = "chunk_002"
DATASET_NAME = "testauthor_golden"
DATASET_VERSION = "7.0"


def _golden_example_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default GoldenExample kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "id": EXAMPLE_ID_001,
        "question": QUESTION_001,
        "language": "en",
        "expected_chunk_ids": [CHUNK_001, CHUNK_002],
    }
    defaults.update(overrides)
    return defaults


def _golden_dataset_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default GoldenDataset kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "name": DATASET_NAME,
        "version": DATASET_VERSION,
        "created_date": "2029-05-09",
        "description": "test dataset for evaluation",
        "examples": [],
    }
    defaults.update(overrides)
    return defaults


def _chat_response_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default ChatResponse kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "text": "Sample response.",
        "retrieved_passage_ids": [CHUNK_001, CHUNK_002],
        "retrieved_contexts": ["Context 1", "Context 2"],
        "retrieved_source_titles": ["Source A"],
        "language": "en",
    }
    defaults.update(overrides)
    return defaults


def _metric_result_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default MetricResult kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "name": "test_metric",
        "score": 0.9,
        "details": {},
    }
    defaults.update(overrides)
    return defaults


class TestRunEval:
    """Tests for run_eval() function."""

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_returns_eval_run(
        self, mock_registry: Mock
    ) -> None:
        """Run eval with mocked chain returns EvalRun with correct structure."""
        # Arrange
        mock_compute = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name="retrieval_relevance", score=1.0
        )))
        metric_spec = MetricSpec(
            name="retrieval_relevance",
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec]))

        example = GoldenExample(**_golden_example_kwargs())
        dataset = GoldenDataset(**_golden_dataset_kwargs(examples=[example]))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        # Import here to avoid circular dependency during collection
        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert
        assert isinstance(result, EvalRun)
        assert result.dataset_version == DATASET_VERSION
        assert result.dataset_name == DATASET_NAME
        assert len(result.example_results) == 1
        assert "overall" in result.aggregate_scores
        assert 0.0 <= result.overall_pass_rate <= 1.0

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_invokes_chain_per_example(
        self, mock_registry: Mock
    ) -> None:
        """Chain.invoke called once per example."""
        # Arrange
        mock_compute = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name="retrieval_relevance", score=0.8
        )))
        metric_spec = MetricSpec(
            name="retrieval_relevance",
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec]))

        example1 = GoldenExample(**_golden_example_kwargs(id=EXAMPLE_ID_001))
        example2 = GoldenExample(**_golden_example_kwargs(id=EXAMPLE_ID_002))
        example3 = GoldenExample(**_golden_example_kwargs(id=EXAMPLE_ID_003))
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            examples=[example1, example2, example3]
        ))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        run_eval(mock_chain, dataset)

        # Assert
        assert mock_chain.invoke.call_count == 3

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_applies_language_specific_metrics(
        self, mock_registry: Mock
    ) -> None:
        """All languages are processed.
        This test will be more meaningful when faithfulness metrics are added.
        """
        mock_compute = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name="retrieval_relevance", score=0.9
        )))
        metric_spec = MetricSpec(
            name="retrieval_relevance",
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec]))

        example_en_1 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_001, language="en", question=QUESTION_001
        ))
        example_fr_1 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_002, language="fr", question=QUESTION_002
        ))
        example_en_2 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_003, language="en", question=QUESTION_003
        ))
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            examples=[example_en_1, example_fr_1, example_en_2]
        ))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert all languages processed and recorded
        assert len(result.example_results) == 3
        assert result.example_results[0].language == "en"
        assert result.example_results[1].language == "fr"
        assert result.example_results[2].language == "en"

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_aggregates_scores(
        self, mock_registry: Mock
    ) -> None:
        """Aggregate scores are present."""
        # Arrange
        mock_compute = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name="retrieval_relevance", score=0.85
        )))
        metric_spec = MetricSpec(
            name="retrieval_relevance",
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec]))

        example_en_1 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_001, language="en", question=QUESTION_001
        ))
        example_fr_1 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_002, language="fr", question=QUESTION_002
        ))
        example_en_2 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_003, language="en", question=QUESTION_003
        ))
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            examples=[example_en_1, example_fr_1, example_en_2]
        ))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert; cross_language will be added when appropriate metrics are added
        assert "overall" in result.aggregate_scores # No assertion on the value returned because that value is mocked
        assert "by_language" in result.aggregate_scores
        assert "en" in result.aggregate_scores["by_language"]
        assert "fr" in result.aggregate_scores["by_language"]

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_calculates_pass_rate(
        self, mock_registry: Mock
    ) -> None:
        """Calculates overall pass rate correctly.
        e.g. 2 of 3 examples pass the threshold → overall_pass_rate ≈ 0.67."""
        # Arrange - return different scores for different examples
        scores = [0.9, 0.5, 0.85]  # 2 pass (>= 0.8), 1 fail
        mock_compute = Mock(side_effect=[
            MetricResult(**_metric_result_kwargs(name="retrieval_relevance", score=s))
            for s in scores
        ])

        # Create MetricSpec once and reuse it
        metric_spec = MetricSpec(
            name="retrieval_relevance",
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )

        # __iter__ must return a new iterator each time it's called
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec]))

        example1 = GoldenExample(**_golden_example_kwargs(id=EXAMPLE_ID_001))
        example2 = GoldenExample(**_golden_example_kwargs(id=EXAMPLE_ID_002))
        example3 = GoldenExample(**_golden_example_kwargs(id=EXAMPLE_ID_003))
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            examples=[example1, example2, example3]
        ))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert
        assert result.overall_pass_rate == pytest.approx(0.667, abs=0.01)
        assert result.example_results[0].passed is True # 0.9 >= 0.8 -> pass
        assert result.example_results[1].passed is False # 0.5 < 0.8 -> fail
        assert result.example_results[2].passed is True # 0.85 >= 0.8 -> pass

    @patch("src.eval.runner._get_system_version")
    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_captures_system_version(
        self, mock_registry: Mock, mock_get_system_version: Mock
    ) -> None:
        """system_version has chat_model, commit fields."""
        # Arrange
        chat_model: str = "test-model-v2"
        commit: str = "abc123def456"
        mock_compute = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name="retrieval_relevance", score=1.0
        )))
        metric_spec = MetricSpec(
            name="retrieval_relevance",
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec]))

        mock_get_system_version.return_value = {
            "chat_model": chat_model,
            "commit": commit,
        }

        example = GoldenExample(**_golden_example_kwargs())
        dataset = GoldenDataset(**_golden_dataset_kwargs(examples=[example]))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert
        #  run_eval() calls _get_system_version(); if it didn't, the mock wouldn't be hit
        #  run_eval() uses the return value instead of ignoring it
        #  run_eval() assigns it to the correct field
        #  The values are correctly nested; result.system_version["chat_model"], not result["chat_model"]
        assert "chat_model" in result.system_version
        assert result.system_version["chat_model"] == chat_model
        assert "commit" in result.system_version
        assert result.system_version["commit"] == commit
