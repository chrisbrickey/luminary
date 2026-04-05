"""Integration tests for eval runner.

This file tests the integration of real components:
- Real, registered metrics via METRIC_REGISTRY
- Real runner (run_eval)
- Real schemas (GoldenDataset, GoldenExample, EvalRun)
- Mock chain (to avoid network calls)

The goal is to verify that all components wire together correctly in the full eval pipeline.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.runnables import Runnable

from src.eval.metrics.base import METRIC_REGISTRY, is_metric_applicable
from src.eval.runner import run_eval
from src.schemas.chat import ChatResponse
from src.schemas.eval import GoldenDataset, GoldenExample


# -- Test constants ---

EXAMPLE_ID_EN_001 = "example_en_001"
EXAMPLE_ID_EN_002 = "example_en_002"
EXAMPLE_ID_FR_001 = "example_fr_001"
EXAMPLE_ID_FR_002 = "example_fr_002"

QUESTION_EN_001 = "What is tolerance?"
QUESTION_EN_002 = "What is reason?"
QUESTION_FR_001 = "Qu'est-ce que la tolérance?"
QUESTION_FR_002 = "Qu'est-ce que la raison?"

CHUNK_001 = "chunk_001"
CHUNK_002 = "chunk_002"
CHUNK_003 = "chunk_003"
CHUNK_004 = "chunk_004"

DATASET_NAME = "testauthor_golden"
DATASET_VERSION = "8.0"


# -- Helper functions ---


def _golden_example_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default GoldenExample kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "id": EXAMPLE_ID_EN_001,
        "question": QUESTION_EN_001,
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
        "created_date": "2029-05-10",
        "description": "test dataset for integration testing",
        "examples": [],
    }
    defaults.update(overrides)
    return defaults


def _chat_response_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default ChatResponse kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "text": "Sample response text.",
        "retrieved_passage_ids": [CHUNK_001, CHUNK_002],
        "retrieved_contexts": ["Context 1", "Context 2"],
        "retrieved_source_titles": ["Source A"],
        "language": "en",
    }
    defaults.update(overrides)
    return defaults


# -- Integration tests ---


def test_eval_runner_end_to_end() -> None:
    """Integration test: wire real metric functions + mocked chain + real runner → verify full pipeline.

    This test verifies:
    1. All applicable metrics from METRIC_REGISTRY are computed for each example
    2. Mock chain returns ChatResponse objects
    3. Real runner (run_eval) processes examples and computes metrics
    4. EvalRun contains expected structure (example_results, aggregate_scores, overall_pass_rate)
    5. Pass/fail logic is consistent: examples pass iff all metrics meet thresholds
    6. Overall pass rate matches actual count of passed examples

    This test is fully dynamic and adapts to any metrics registered in METRIC_REGISTRY.
    """
    # Arrange: Create golden dataset with 3 examples
    # Test data is designed to produce varied metric scores:
    # - 2 examples with perfect matches (expected == retrieved)
    # - 1 example with no overlap (expected ∩ retrieved = ∅)
    example_pass_1 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_EN_001,
            question=QUESTION_EN_001,
            language="en",
            expected_chunk_ids=[CHUNK_001, CHUNK_002],
        )
    )
    example_pass_2 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_EN_002,
            question=QUESTION_EN_002,
            language="en",
            expected_chunk_ids=[CHUNK_003],
        )
    )
    example_fail = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_FR_001,
            question=QUESTION_FR_001,
            language="fr",
            expected_chunk_ids=[CHUNK_001, CHUNK_002],
        )
    )

    dataset = GoldenDataset(
        **_golden_dataset_kwargs(
            examples=[example_pass_1, example_pass_2, example_fail]
        )
    )

    # Arrange: Create mock chain that returns different responses per example
    mock_chain = Mock(spec=Runnable)

    # Define responses: perfect match for first 2, no match for 3rd
    response_pass_1 = ChatResponse(
        **_chat_response_kwargs(
            text="Response about tolerance.",
            retrieved_passage_ids=[CHUNK_001, CHUNK_002],  # Perfect match
            retrieved_contexts=["Context 1", "Context 2"],
            retrieved_source_titles=["Source A"],
            language="en",
        )
    )
    response_pass_2 = ChatResponse(
        **_chat_response_kwargs(
            text="Response about reason.",
            retrieved_passage_ids=[CHUNK_003],  # Perfect match
            retrieved_contexts=["Context 3"],
            retrieved_source_titles=["Source B"],
            language="en",
        )
    )
    response_fail = ChatResponse(
        **_chat_response_kwargs(
            text="Réponse sur la tolérance.",
            retrieved_passage_ids=[CHUNK_003, CHUNK_004],  # No overlap with expected
            retrieved_contexts=["Contexte 3", "Contexte 4"],
            retrieved_source_titles=["Source C"],
            language="fr",
        )
    )

    mock_chain.invoke.side_effect = [response_pass_1, response_pass_2, response_fail]

    # Act: Run eval with real runner + real metrics + mock chain
    result = run_eval(mock_chain, dataset)

    # Assert: Verify EvalRun structure
    assert result.dataset_name == DATASET_NAME
    assert result.dataset_version == DATASET_VERSION
    assert len(result.example_results) == 3

    # Assert: Verify chain was invoked for each example
    assert mock_chain.invoke.call_count == 3

    # Assert: Verify all applicable metrics from METRIC_REGISTRY are computed
    # Get the first example and response to check applicability
    example_for_check = dataset.examples[0]
    response_for_check = response_pass_1

    # Determine which metrics should be applicable to our test data
    applicable_metric_names = {
        spec.name
        for spec in METRIC_REGISTRY
        if is_metric_applicable(spec, example_for_check, response_for_check)
    }

    # Verify each example has all applicable metrics
    for example_result in result.example_results:
        assert len(example_result.metrics) > 0
        computed_metric_names = {m.name for m in example_result.metrics}

        # All applicable metrics should be computed
        # (Note: some metrics might not apply to all examples due to language constraints)
        # So we check that at least the metrics we expect are present
        for metric_name in applicable_metric_names:
            # Check if this metric applies to this specific example
            example_idx = result.example_results.index(example_result)
            original_example = dataset.examples[example_idx]
            response_idx = example_idx
            original_response = [response_pass_1, response_pass_2, response_fail][response_idx]

            # Find the metric spec
            metric_spec = next(spec for spec in METRIC_REGISTRY if spec.name == metric_name)

            if is_metric_applicable(metric_spec, original_example, original_response):
                assert metric_name in computed_metric_names, (
                    f"Expected metric '{metric_name}' to be computed for example {example_result.id}"
                )

    # Assert: Verify metric thresholds are persisted for all registered metrics
    for spec in METRIC_REGISTRY:
        assert spec.name in result.effective_thresholds, (
            f"Expected metric '{spec.name}' to have a threshold in effective_thresholds"
        )
        assert result.effective_thresholds[spec.name] == spec.default_threshold

    # Assert: Verify pass/fail logic is consistent with scores and thresholds
    # An example passes if ALL its metrics meet their thresholds
    for example_result in result.example_results:
        for metric in example_result.metrics:
            threshold = result.effective_thresholds[metric.name]
            if example_result.passed:
                # If example passed, all metrics must be >= threshold
                assert metric.score >= threshold, (
                    f"Example {example_result.id} marked as passed but "
                    f"metric '{metric.name}' scored {metric.score} < threshold {threshold}"
                )

        # If example failed, verify at least one metric was below threshold
        if not example_result.passed:
            failing_metrics = [
                m for m in example_result.metrics
                if m.score < result.effective_thresholds[m.name]
            ]
            assert len(failing_metrics) > 0, (
                f"Example {example_result.id} marked as failed but "
                f"all metrics meet thresholds"
            )

    # Assert: Verify overall pass rate matches count of passed examples
    passed_count = sum(1 for ex in result.example_results if ex.passed)
    expected_pass_rate = passed_count / len(result.example_results)
    assert result.overall_pass_rate == pytest.approx(expected_pass_rate, abs=0.001)

    # Assert: Verify aggregate scores structure
    assert "overall" in result.aggregate_scores
    assert "by_language" in result.aggregate_scores

    # Assert: Verify all applicable metrics appear in aggregate scores
    for metric_name in applicable_metric_names:
        assert metric_name in result.aggregate_scores["overall"], (
            f"Expected metric '{metric_name}' in overall aggregate scores"
        )

    # Assert: Verify system version captured
    assert "chat_model" in result.system_version
    assert "commit" in result.system_version
    assert "timestamp" in result.system_version


def test_eval_runner_processes_multilingual_dataset() -> None:
    """Integration test: dataset with FR and EN examples → both processed, cross-language metrics computed.

    This test verifies:
    1. Examples in multiple languages (en, fr) are all processed
    2. Language-specific responses are matched correctly
    3. Aggregate scores are computed by language (by_language breakdown)
    4. Language constraints on metrics are respected (metrics with languages=None apply to all)
    5. Pass/fail logic is consistent with metric scores and thresholds

    This test is fully dynamic and adapts to any metrics registered in METRIC_REGISTRY.
    """
    # Arrange: Create multilingual dataset
    # - 2 English examples
    # - 2 French examples
    example_en_1 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_EN_001,
            question=QUESTION_EN_001,
            language="en",
            expected_chunk_ids=[CHUNK_001],
        )
    )
    example_en_2 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_EN_002,
            question=QUESTION_EN_002,
            language="en",
            expected_chunk_ids=[CHUNK_002],
        )
    )
    example_fr_1 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_FR_001,
            question=QUESTION_FR_001,
            language="fr",
            expected_chunk_ids=[CHUNK_003],
        )
    )
    example_fr_2 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_FR_002,
            question=QUESTION_FR_002,
            language="fr",
            expected_chunk_ids=[CHUNK_004],
        )
    )

    dataset = GoldenDataset(
        **_golden_dataset_kwargs(
            examples=[example_en_1, example_en_2, example_fr_1, example_fr_2]
        )
    )

    # Arrange: Create mock chain with multilingual responses
    mock_chain = Mock(spec=Runnable)

    response_en_1 = ChatResponse(
        **_chat_response_kwargs(
            text="Response about tolerance.",
            retrieved_passage_ids=[CHUNK_001],
            retrieved_contexts=["Context 1"],
            retrieved_source_titles=["Source A"],
            language="en",
        )
    )
    response_en_2 = ChatResponse(
        **_chat_response_kwargs(
            text="Response about reason.",
            retrieved_passage_ids=[CHUNK_002],
            retrieved_contexts=["Context 2"],
            retrieved_source_titles=["Source B"],
            language="en",
        )
    )
    response_fr_1 = ChatResponse(
        **_chat_response_kwargs(
            text="Réponse sur la tolérance.",
            retrieved_passage_ids=[CHUNK_003],
            retrieved_contexts=["Contexte 3"],
            retrieved_source_titles=["Source C"],
            language="fr",
        )
    )
    response_fr_2 = ChatResponse(
        **_chat_response_kwargs(
            text="Réponse sur la raison.",
            retrieved_passage_ids=[CHUNK_004],
            retrieved_contexts=["Contexte 4"],
            retrieved_source_titles=["Source D"],
            language="fr",
        )
    )

    mock_chain.invoke.side_effect = [
        response_en_1,
        response_en_2,
        response_fr_1,
        response_fr_2,
    ]

    # Act: Run eval with multilingual dataset
    result = run_eval(mock_chain, dataset)

    # Assert: Verify all examples processed
    assert len(result.example_results) == 4
    assert mock_chain.invoke.call_count == 4

    # Assert: Verify languages are preserved in example results
    assert result.example_results[0].language == "en"
    assert result.example_results[1].language == "en"
    assert result.example_results[2].language == "fr"
    assert result.example_results[3].language == "fr"

    # Assert: Verify pass/fail logic is consistent with scores and thresholds
    for example_result in result.example_results:
        for metric in example_result.metrics:
            threshold = result.effective_thresholds[metric.name]
            if example_result.passed:
                # If example passed, all metrics must be >= threshold
                assert metric.score >= threshold, (
                    f"Example {example_result.id} marked as passed but "
                    f"metric '{metric.name}' scored {metric.score} < threshold {threshold}"
                )

        # If example failed, verify at least one metric was below threshold
        if not example_result.passed:
            failing_metrics = [
                m for m in example_result.metrics
                if m.score < result.effective_thresholds[m.name]
            ]
            assert len(failing_metrics) > 0, (
                f"Example {example_result.id} marked as failed but "
                f"all metrics meet thresholds"
            )

    # Assert: Verify overall pass rate matches count of passed examples
    passed_count = sum(1 for ex in result.example_results if ex.passed)
    expected_pass_rate = passed_count / len(result.example_results)
    assert result.overall_pass_rate == pytest.approx(expected_pass_rate, abs=0.001)

    # Assert: Verify aggregate scores by language
    assert "by_language" in result.aggregate_scores
    assert "en" in result.aggregate_scores["by_language"]
    assert "fr" in result.aggregate_scores["by_language"]

    # Assert: Verify all applicable metrics from METRIC_REGISTRY are computed by language
    # Determine which metrics should be applicable to our test data
    example_for_check = dataset.examples[0]
    response_for_check = response_en_1

    applicable_metric_names = {
        spec.name
        for spec in METRIC_REGISTRY
        if is_metric_applicable(spec, example_for_check, response_for_check)
    }

    # Check that each applicable metric appears in both language aggregates
    for metric_name in applicable_metric_names:
        # Find the metric spec to check language constraints
        metric_spec = next(spec for spec in METRIC_REGISTRY if spec.name == metric_name)

        # If metric applies to all languages or specifically to 'en'/'fr', check it's present
        if metric_spec.languages is None or "en" in metric_spec.languages:
            assert metric_name in result.aggregate_scores["by_language"]["en"], (
                f"Expected metric '{metric_name}' in English aggregate scores"
            )

        if metric_spec.languages is None or "fr" in metric_spec.languages:
            assert metric_name in result.aggregate_scores["by_language"]["fr"], (
                f"Expected metric '{metric_name}' in French aggregate scores"
            )

