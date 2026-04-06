"""Tests for src/eval/runner.py — evaluation runner."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.runnables import Runnable

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.eval.metrics.base import MetricSpec, FALLBACK_THRESHOLD
from src.schemas.chat import ChatResponse
from src.schemas.eval import EvalRun, GoldenDataset, GoldenExample, MetricResult


# -- Shared test constants and methods ---

# Example identifiers
EXAMPLE_ID_001 = "example_001"
EXAMPLE_ID_002 = "example_002"
EXAMPLE_ID_003 = "example_003"
QUESTION_001 = "What is tolerance?"
QUESTION_002 = "Qu'est-ce que la tolérance?"
QUESTION_003 = "What is reason?"

# Chunk and dataset identifiers
CHUNK_001 = "chunk_001"
CHUNK_002 = "chunk_002"
DATASET_NAME = "golden_testauthor"
DATASET_VERSION = "7.0"

# Metric names
METRIC_NAME = "test_metric"
METRIC_NAME_2 = "test_metric_2"


def _golden_example_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default GoldenExample kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "id": EXAMPLE_ID_001,
        "question": QUESTION_001,
        "language": ENGLISH_ISO_CODE,
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
        "language": ENGLISH_ISO_CODE,
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
            name=METRIC_NAME, score=1.0
        )))
        metric_spec = MetricSpec(
            name=METRIC_NAME,
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
        assert METRIC_NAME in result.effective_thresholds
        assert result.effective_thresholds[METRIC_NAME] == FALLBACK_THRESHOLD

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_invokes_chain_per_example(
        self, mock_registry: Mock
    ) -> None:
        """Chain.invoke called once per example."""
        # Arrange
        mock_compute = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME, score=0.8
        )))
        metric_spec = MetricSpec(
            name=METRIC_NAME,
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
            name=METRIC_NAME, score=0.9
        )))
        metric_spec = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec]))

        example_en_1 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_001, language=ENGLISH_ISO_CODE, question=QUESTION_001
        ))
        example_fr_1 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_002, language=FRENCH_ISO_CODE, question=QUESTION_002
        ))
        example_en_2 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_003, language=ENGLISH_ISO_CODE, question=QUESTION_003
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
        assert result.example_results[0].language == ENGLISH_ISO_CODE
        assert result.example_results[1].language == FRENCH_ISO_CODE
        assert result.example_results[2].language == ENGLISH_ISO_CODE

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_aggregates_scores(
        self, mock_registry: Mock
    ) -> None:
        """Aggregate scores are present."""
        # Arrange
        mock_compute = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME, score=0.85
        )))
        metric_spec = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec]))

        example_en_1 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_001, language=ENGLISH_ISO_CODE, question=QUESTION_001
        ))
        example_fr_1 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_002, language=FRENCH_ISO_CODE, question=QUESTION_002
        ))
        example_en_2 = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_003, language=ENGLISH_ISO_CODE, question=QUESTION_003
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
        assert ENGLISH_ISO_CODE in result.aggregate_scores["by_language"]
        assert FRENCH_ISO_CODE in result.aggregate_scores["by_language"]

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_calculates_pass_rate(
        self, mock_registry: Mock
    ) -> None:
        """Calculates overall pass rate correctly.
        e.g. 2 of 3 examples pass the threshold → overall_pass_rate ≈ 0.67."""
        # Arrange - return different scores for different examples
        scores = [0.9, 0.5, 0.85]  # 2 pass (>= 0.8), 1 fail
        mock_compute = Mock(side_effect=[
            MetricResult(**_metric_result_kwargs(name=METRIC_NAME, score=s))
            for s in scores
        ])

        # Create MetricSpec once and reuse it
        metric_spec = MetricSpec(
            name=METRIC_NAME,
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
            name=METRIC_NAME, score=1.0
        )))
        metric_spec = MetricSpec(
            name=METRIC_NAME,
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

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_uses_custom_metric_thresholds(
        self, mock_registry: Mock
    ) -> None:
        """Custom metric thresholds override defaults and are recorded in EvalRun."""
        # Arrange - return different scores for different examples
        scores = [0.75, 0.65, 0.85]  # With threshold 0.7: 2 pass, 1 fail
        mock_compute = Mock(side_effect=[
            MetricResult(**_metric_result_kwargs(name=METRIC_NAME, score=s))
            for s in scores
        ])

        metric_spec = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
            default_threshold=0.8,  # Default is 0.8
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

        # Act - override threshold to 0.7 (lower than 0.8 of FALLBACK_THRESHOLD)
        custom_thresholds = {METRIC_NAME: 0.7}
        result = run_eval(mock_chain, dataset, override_thresholds=custom_thresholds)

        # Assert - verify custom threshold is used for pass/fail
        # With threshold 0.7: 0.75 passes, 0.65 fails, 0.85 passes
        assert result.example_results[0].passed is True  # 0.75 >= 0.7 -> pass
        assert result.example_results[1].passed is False  # 0.65 < 0.7 -> fail
        assert result.example_results[2].passed is True  # 0.85 >= 0.7 -> pass
        assert result.overall_pass_rate == pytest.approx(0.667, abs=0.01)

        # Assert - verify custom threshold is recorded in EvalRun
        assert result.effective_thresholds[METRIC_NAME] == 0.7

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_computes_all_registered_metrics(
        self, mock_registry: Mock
    ) -> None:
        """All registered metrics are computed for each example."""
        # Arrange - create two metrics
        mock_compute_1 = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME, score=0.9
        )))
        mock_compute_2 = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME_2, score=0.85
        )))

        metric_spec_1 = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute_1,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        metric_spec_2 = MetricSpec(
            name=METRIC_NAME_2,
            compute=mock_compute_2,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec_1, metric_spec_2]))

        example1 = GoldenExample(**_golden_example_kwargs(id=EXAMPLE_ID_001))
        example2 = GoldenExample(**_golden_example_kwargs(id=EXAMPLE_ID_002))
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            examples=[example1, example2]
        ))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert - both metrics computed for each example
        assert mock_compute_1.call_count == 2  # Called once per example
        assert mock_compute_2.call_count == 2  # Called once per example

        # Assert - both metric results appear in each example's results
        for example_result in result.example_results:
            metric_names = [mr.name for mr in example_result.metrics]
            assert METRIC_NAME in metric_names
            assert METRIC_NAME_2 in metric_names

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_aggregates_multiple_metrics(
        self, mock_registry: Mock
    ) -> None:
        """Aggregate scores include all metrics."""
        # Arrange - create two metrics with different scores
        mock_compute_1 = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME, score=0.9
        )))
        mock_compute_2 = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME_2, score=0.7
        )))

        metric_spec_1 = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute_1,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        metric_spec_2 = MetricSpec(
            name=METRIC_NAME_2,
            compute=mock_compute_2,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec_1, metric_spec_2]))

        example = GoldenExample(**_golden_example_kwargs())
        dataset = GoldenDataset(**_golden_dataset_kwargs(examples=[example]))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert - both metrics appear in aggregate scores
        assert "overall" in result.aggregate_scores
        overall_scores = result.aggregate_scores["overall"]
        assert METRIC_NAME in overall_scores
        assert METRIC_NAME_2 in overall_scores

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_pass_fail_requires_all_metrics_pass(
        self, mock_registry: Mock
    ) -> None:
        """Example passes only if ALL metrics pass their thresholds."""
        # Arrange - create two metrics with different scores for different examples
        # Example 1: both pass (0.9, 0.85) >= 0.8
        # Example 2: one passes, one fails (0.9, 0.7) - one < 0.8
        # Example 3: both fail (0.5, 0.6) < 0.8
        scores_metric_1 = [0.9, 0.9, 0.5]
        scores_metric_2 = [0.85, 0.7, 0.6]

        mock_compute_1 = Mock(side_effect=[
            MetricResult(**_metric_result_kwargs(name=METRIC_NAME, score=s))
            for s in scores_metric_1
        ])
        mock_compute_2 = Mock(side_effect=[
            MetricResult(**_metric_result_kwargs(name=METRIC_NAME_2, score=s))
            for s in scores_metric_2
        ])

        metric_spec_1 = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute_1,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        metric_spec_2 = MetricSpec(
            name=METRIC_NAME_2,
            compute=mock_compute_2,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec_1, metric_spec_2]))

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

        # Assert - only example 1 passes (both metrics >= 0.8)
        assert result.example_results[0].passed is True   # Both pass: 0.9, 0.85
        assert result.example_results[1].passed is False  # One fails: 0.9, 0.7
        assert result.example_results[2].passed is False  # Both fail: 0.5, 0.6
        assert result.overall_pass_rate == pytest.approx(0.333, abs=0.01)  # 1/3 pass

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_records_thresholds_for_all_metrics(
        self, mock_registry: Mock
    ) -> None:
        """Effective thresholds recorded for all metrics."""
        # Arrange - create two metrics with different default thresholds
        mock_compute_1 = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME, score=0.9
        )))
        mock_compute_2 = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME_2, score=0.85
        )))

        metric_spec_1 = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute_1,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
            default_threshold=0.8,
        )
        metric_spec_2 = MetricSpec(
            name=METRIC_NAME_2,
            compute=mock_compute_2,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,
            default_threshold=0.75,
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec_1, metric_spec_2]))

        example = GoldenExample(**_golden_example_kwargs())
        dataset = GoldenDataset(**_golden_dataset_kwargs(examples=[example]))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert - both thresholds recorded
        assert METRIC_NAME in result.effective_thresholds
        assert METRIC_NAME_2 in result.effective_thresholds
        assert result.effective_thresholds[METRIC_NAME] == 0.8
        assert result.effective_thresholds[METRIC_NAME_2] == 0.75

    @patch("src.eval.runner.METRIC_REGISTRY")
    def test_run_eval_filters_metrics_by_language(
        self, mock_registry: Mock
    ) -> None:
        """Metrics with language restrictions only apply to matching examples."""
        # Arrange - create two metrics
        # Metric 1: English-only
        # Metric 2: All languages (None)
        mock_compute_1 = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME, score=0.9
        )))
        mock_compute_2 = Mock(return_value=MetricResult(**_metric_result_kwargs(
            name=METRIC_NAME_2, score=0.85
        )))

        metric_spec_1 = MetricSpec(
            name=METRIC_NAME,
            compute=mock_compute_1,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=[ENGLISH_ISO_CODE],  # English only
        )
        metric_spec_2 = MetricSpec(
            name=METRIC_NAME_2,
            compute=mock_compute_2,
            required_example_fields=set(),
            required_response_fields=set(),
            languages=None,  # All languages
        )
        mock_registry.__iter__ = Mock(side_effect=lambda: iter([metric_spec_1, metric_spec_2]))

        example_en = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_001, language=ENGLISH_ISO_CODE, question=QUESTION_001
        ))
        example_fr = GoldenExample(**_golden_example_kwargs(
            id=EXAMPLE_ID_002, language=FRENCH_ISO_CODE, question=QUESTION_002
        ))
        dataset = GoldenDataset(**_golden_dataset_kwargs(
            examples=[example_en, example_fr]
        ))

        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(**_chat_response_kwargs())

        from src.eval.runner import run_eval

        # Act
        result = run_eval(mock_chain, dataset)

        # Assert - English example has both metrics
        en_result = result.example_results[0]
        en_metric_names = [mr.name for mr in en_result.metrics]
        assert METRIC_NAME in en_metric_names     # English-only metric applies
        assert METRIC_NAME_2 in en_metric_names   # All-language metric applies

        # Assert - French example has only the all-language metric
        fr_result = result.example_results[1]
        fr_metric_names = [mr.name for mr in fr_result.metrics]
        assert METRIC_NAME not in fr_metric_names  # English-only metric skipped
        assert METRIC_NAME_2 in fr_metric_names    # All-language metric applies
