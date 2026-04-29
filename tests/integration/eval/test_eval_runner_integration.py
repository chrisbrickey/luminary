"""Integration tests for eval runner and CLI.

This file tests the integration of real components:
- Real runner (run_eval)
- Real, registered metrics via METRIC_REGISTRY
- Real schemas (GoldenDataset, GoldenExample, EvalRun)
- Real production configuration (DEFAULT_AUTHOR) unless testing multi-author scenarios
- Real file I/O (reading golden datasets, writing artifacts)
- Mock chain (to avoid network calls)

The goal is to verify that all components wire together correctly in the full eval pipeline,
including CLI workflows, file operations, and error handling.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.runnables import Runnable

from src.configs.authors import DEFAULT_AUTHOR
from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from src.eval.metrics.base import METRIC_REGISTRY, is_metric_applicable
from src.eval.runner import get_system_version, run_eval
from src.schemas.chat import ChatResponse
from src.schemas.eval import EvalRun, GoldenDataset, GoldenExample, SystemVersion
from tests.conftest import FakeChatModel


# -- Test constants ---

# Example identifiers
EXAMPLE_ID_EN_001 = "example_en_001"
EXAMPLE_ID_EN_002 = "example_en_002"
EXAMPLE_ID_FR_001 = "example_fr_001"
EXAMPLE_ID_FR_002 = "example_fr_002"

# Questions
QUESTION_EN_001 = "What is tolerance?"
QUESTION_EN_002 = "What is reason?"
QUESTION_FR_001 = "Qu'est-ce que la tolérance?"
QUESTION_FR_002 = "Qu'est-ce que la raison?"

# Chunk identifiers
CHUNK_001 = "chunk_001"
CHUNK_002 = "chunk_002"
CHUNK_003 = "chunk_003"
CHUNK_004 = "chunk_004"

# Dataset metadata
DATASET_SCOPE = "persona"
DATASET_VERSION = "8.0"
DATASET_DATE = "2029-05-10"
DATASET_IDENTIFIER = f"{DATASET_SCOPE}_{DEFAULT_AUTHOR}_v{DATASET_VERSION}_{DATASET_DATE}"

# Contexts and sources (for ChatResponse)
CONTEXT_001 = "Context about tolerance and enlightenment values."
CONTEXT_002 = "Context about reason and scientific method."
CONTEXT_003 = "Context about progress and human advancement."
CONTEXT_004 = "Context about liberty and natural rights."
SOURCE_TITLE_A = "Philosophical Essays, Page 12"
SOURCE_TITLE_B = "Letters on Tolerance, Page 5"


# -- Helper functions ---


def _golden_example_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default GoldenExample kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "id": EXAMPLE_ID_EN_001,
        "question": QUESTION_EN_001,
        "author": DEFAULT_AUTHOR,
        "language": ENGLISH_ISO_CODE,
        "expected_chunk_ids": [CHUNK_001, CHUNK_002],
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
        "retrieved_contexts": [CONTEXT_001, CONTEXT_002],
        "retrieved_source_titles": [SOURCE_TITLE_A],
        "language": ENGLISH_ISO_CODE,
    }
    defaults.update(overrides)
    return defaults


def _create_golden_dataset_file(
    directory: Path,
    scope: str = DATASET_SCOPE,
    authors: list[str] | None = None,
    version: str = DATASET_VERSION,
    created_date: str = DATASET_DATE,
    num_examples: int = 1,
) -> Path:
    """Create a golden dataset JSON file in the specified directory.

    Args:
        directory: Directory to create the file in
        scope: Dataset scope
        authors: List of authors (defaults to [DEFAULT_AUTHOR])
        version: Dataset version
        created_date: ISO date
        num_examples: Number of examples to include

    Returns:
        Path to the created JSON file
    """
    if authors is None:
        authors = [DEFAULT_AUTHOR]

    # Create examples
    examples = []
    for i in range(num_examples):
        examples.append(
            GoldenExample(
                **_golden_example_kwargs(
                    id=f"example_{i:03d}",
                    question=QUESTION_EN_001 if i % 2 == 0 else QUESTION_FR_001,
                    author=authors[i % len(authors)],
                    language=ENGLISH_ISO_CODE if i % 2 == 0 else FRENCH_ISO_CODE,
                    expected_chunk_ids=[CHUNK_001, CHUNK_002],
                )
            )
        )

    # Create dataset
    dataset = GoldenDataset(
        **_golden_dataset_kwargs(
            scope=scope,
            authors=sorted(authors),
            version=version,
            created_date=created_date,
            examples=examples,
        )
    )

    # Write to file with standard naming convention
    authors_str = "_".join(sorted(authors))
    filename = f"{scope}_{authors_str}_v{version}_{created_date}.json"
    filepath = directory / filename
    filepath.write_text(dataset.model_dump_json(indent=2))

    return filepath


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
            language=ENGLISH_ISO_CODE,
            expected_chunk_ids=[CHUNK_001, CHUNK_002],
        )
    )
    example_pass_2 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_EN_002,
            question=QUESTION_EN_002,
            language=ENGLISH_ISO_CODE,
            expected_chunk_ids=[CHUNK_003],
        )
    )
    example_fail = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_FR_001,
            question=QUESTION_FR_001,
            language=FRENCH_ISO_CODE,
            expected_chunk_ids=[CHUNK_001, CHUNK_002],
        )
    )

    dataset = GoldenDataset(
        **_golden_dataset_kwargs(
            authors=[DEFAULT_AUTHOR],
            examples=[example_pass_1, example_pass_2, example_fail]
        )
    )

    # Arrange: Create mock chain that returns different responses per example
    mock_chain = Mock(spec=Runnable)
    chains = {DEFAULT_AUTHOR: mock_chain}

    # Define responses: perfect match for first 2, no match for 3rd
    response_pass_1 = ChatResponse(
        **_chat_response_kwargs(
            text="Response about tolerance.",
            retrieved_passage_ids=[CHUNK_001, CHUNK_002],  # Perfect match
            retrieved_contexts=[CONTEXT_001, CONTEXT_002],
            retrieved_source_titles=[SOURCE_TITLE_A],
            language=ENGLISH_ISO_CODE,
        )
    )
    response_pass_2 = ChatResponse(
        **_chat_response_kwargs(
            text="Response about reason.",
            retrieved_passage_ids=[CHUNK_003],  # Perfect match
            retrieved_contexts=[CONTEXT_003],
            retrieved_source_titles=[SOURCE_TITLE_B],
            language=ENGLISH_ISO_CODE,
        )
    )
    response_fail = ChatResponse(
        **_chat_response_kwargs(
            text="Réponse sur la tolérance.",
            retrieved_passage_ids=[CHUNK_003, CHUNK_004],  # No overlap with expected
            retrieved_contexts=[CONTEXT_003, CONTEXT_004],
            retrieved_source_titles=[SOURCE_TITLE_B],
            language=FRENCH_ISO_CODE,
        )
    )

    mock_chain.invoke.side_effect = [response_pass_1, response_pass_2, response_fail]

    # Act: Run eval with real runner + real metrics + mock chain
    result = run_eval(dataset, chains)

    # Assert: Verify EvalRun correctly extracts metadata from GoldenDataset
    assert result.dataset_scope == dataset.scope
    assert result.dataset_authors == dataset.authors
    assert result.dataset_identifier == dataset.identifier
    assert result.dataset_version == dataset.version
    assert result.dataset_date == dataset.created_date
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
    system_version = result.system_version
    assert isinstance(system_version, SystemVersion)

    for field_name in SystemVersion.model_fields:
        assert getattr(system_version, field_name), (
            f"Expected system_version.{field_name} to be set"
        )


def test_get_system_version_returns_all_fields() -> None:
    """Integration test: get_system_version() wires all constants correctly.

    The unit runner test mocks get_system_version, so this is the only test
    that exercises the real function and verifies every field is populated.
    """
    result = get_system_version()

    assert isinstance(result, SystemVersion)

    for field_name in SystemVersion.model_fields:
        assert getattr(result, field_name), (
            f"Expected system_version.{field_name} to be set"
        )


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
            language=ENGLISH_ISO_CODE,
            expected_chunk_ids=[CHUNK_001],
        )
    )
    example_en_2 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_EN_002,
            question=QUESTION_EN_002,
            language=ENGLISH_ISO_CODE,
            expected_chunk_ids=[CHUNK_002],
        )
    )
    example_fr_1 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_FR_001,
            question=QUESTION_FR_001,
            language=FRENCH_ISO_CODE,
            expected_chunk_ids=[CHUNK_003],
        )
    )
    example_fr_2 = GoldenExample(
        **_golden_example_kwargs(
            id=EXAMPLE_ID_FR_002,
            question=QUESTION_FR_002,
            language=FRENCH_ISO_CODE,
            expected_chunk_ids=[CHUNK_004],
        )
    )

    dataset = GoldenDataset(
        **_golden_dataset_kwargs(
            authors=[DEFAULT_AUTHOR],
            examples=[example_en_1, example_en_2, example_fr_1, example_fr_2]
        )
    )

    # Arrange: Create mock chain with multilingual responses
    mock_chain = Mock(spec=Runnable)
    chains = {DEFAULT_AUTHOR: mock_chain}

    response_en_1 = ChatResponse(
        **_chat_response_kwargs(
            text="Response about tolerance.",
            retrieved_passage_ids=[CHUNK_001],
            retrieved_contexts=[CONTEXT_001],
            retrieved_source_titles=[SOURCE_TITLE_A],
            language=ENGLISH_ISO_CODE,
        )
    )
    response_en_2 = ChatResponse(
        **_chat_response_kwargs(
            text="Response about reason.",
            retrieved_passage_ids=[CHUNK_002],
            retrieved_contexts=[CONTEXT_002],
            retrieved_source_titles=[SOURCE_TITLE_B],
            language=ENGLISH_ISO_CODE,
        )
    )
    response_fr_1 = ChatResponse(
        **_chat_response_kwargs(
            text="Réponse sur la tolérance.",
            retrieved_passage_ids=[CHUNK_003],
            retrieved_contexts=[CONTEXT_003],
            retrieved_source_titles=[SOURCE_TITLE_A],
            language=FRENCH_ISO_CODE,
        )
    )
    response_fr_2 = ChatResponse(
        **_chat_response_kwargs(
            text="Réponse sur la raison.",
            retrieved_passage_ids=[CHUNK_004],
            retrieved_contexts=[CONTEXT_004],
            retrieved_source_titles=[SOURCE_TITLE_B],
            language=FRENCH_ISO_CODE,
        )
    )

    mock_chain.invoke.side_effect = [
        response_en_1,
        response_en_2,
        response_fr_1,
        response_fr_2,
    ]

    # Act: Run eval with multilingual dataset
    result = run_eval(dataset, chains)

    # Assert: Verify all examples processed
    assert len(result.example_results) == 4
    assert mock_chain.invoke.call_count == 4

    # Assert: Verify languages are preserved in example results
    assert result.example_results[0].language == ENGLISH_ISO_CODE
    assert result.example_results[1].language == ENGLISH_ISO_CODE
    assert result.example_results[2].language == FRENCH_ISO_CODE
    assert result.example_results[3].language == FRENCH_ISO_CODE

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
    assert ENGLISH_ISO_CODE in result.aggregate_scores["by_language"]
    assert FRENCH_ISO_CODE in result.aggregate_scores["by_language"]

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
        if metric_spec.languages is None or ENGLISH_ISO_CODE in metric_spec.languages:
            assert metric_name in result.aggregate_scores["by_language"][ENGLISH_ISO_CODE], (
                f"Expected metric '{metric_name}' in English aggregate scores"
            )

        if metric_spec.languages is None or FRENCH_ISO_CODE in metric_spec.languages:
            assert metric_name in result.aggregate_scores["by_language"][FRENCH_ISO_CODE], (
                f"Expected metric '{metric_name}' in French aggregate scores"
            )


class TestCLIIntegration:
    """Integration tests for CLI script (scripts/run_eval.py).

    These tests verify the full CLI workflow including:
    - CLI argument parsing and defaults
    - File I/O (reading golden datasets, writing artifacts)
    - Integration with real eval runner and real metrics
    - Terminal output and error handling
    - Mock chain/LLM to avoid network calls
    """

    def test_cli_auto_discovery_workflow(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI with auto-discovery of golden dataset and artifact creation.

        Verifies:
        - Auto-discovery finds golden dataset in default directory
        - Real eval runner processes examples with real metrics
        - Mock chain avoids network calls
        - Artifact is saved to output directory
        - Summary is printed to stdout
        """
        # Arrange: Create golden dataset in default location
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        golden_path = _create_golden_dataset_file(
            directory=golden_dir,
            num_examples=2,
        )

        # Arrange: Output directory
        output_dir = tmp_path / "runs"

        # Arrange: Mock chain with predictable responses
        mock_chain = Mock(spec=Runnable)
        # Return perfect matches for both examples
        mock_chain.invoke.side_effect = [
            ChatResponse(
                **_chat_response_kwargs(
                    text="Test response 1",
                    retrieved_passage_ids=[CHUNK_001, CHUNK_002],
                    language=ENGLISH_ISO_CODE,
                )
            ),
            ChatResponse(
                **_chat_response_kwargs(
                    text="Test response 2",
                    retrieved_passage_ids=[CHUNK_001, CHUNK_002],
                    language=FRENCH_ISO_CODE,
                )
            ),
        ]

        # Act: Run CLI with auto-discovery (no --golden-path)
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", output_dir), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Artifact file created
        artifact_files = list(output_dir.glob("*.json"))
        assert len(artifact_files) == 1, "Expected exactly one artifact file"

        # Assert: Artifact contains valid EvalRun
        saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
        assert saved_run.dataset_scope == DATASET_SCOPE
        assert saved_run.dataset_authors == [DEFAULT_AUTHOR]
        assert len(saved_run.example_results) == 2

        # Assert: Summary printed to stdout
        captured = capsys.readouterr()
        assert "OVERALL SCORES" in captured.out
        assert "OVERALL PASS RATE" in captured.out

        # Assert: Chain was invoked for each example
        assert mock_chain.invoke.call_count == 2

    def test_cli_explicit_paths_workflow(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
    ) -> None:
        """Test CLI with explicit --golden-path and --output-path.

        Verifies:
        - Explicit golden path loads correct dataset
        - Custom output path is used for artifacts
        - Real eval runner and metrics work correctly
        """
        # Arrange: Create golden dataset at custom location
        custom_golden_dir = tmp_path / "custom" / "golden"
        custom_golden_dir.mkdir(parents=True)
        golden_path = _create_golden_dataset_file(
            directory=custom_golden_dir,
            version="9.5",
            num_examples=1,
        )

        # Arrange: Custom output directory
        custom_output = tmp_path / "custom" / "output"

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(
                retrieved_passage_ids=[CHUNK_001, CHUNK_002],
            )
        )

        # Act: Run CLI with explicit paths
        with patch("sys.argv", [
                "run_eval.py",
                "--golden-path", str(golden_path),
                "--output-path", str(custom_output),
            ]), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Artifact created in custom output directory
        artifact_files = list(custom_output.glob("*.json"))
        assert len(artifact_files) == 1

        # Assert: Artifact reflects custom dataset version
        saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
        assert saved_run.dataset_version == "9.5"

    def test_cli_verbose_logging(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI --verbose flag enables debug logging.

        Verifies:
        - --verbose flag is parsed correctly
        - Debug log messages appear in stderr
        """
        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001, CHUNK_002])
        )

        # Act: Run with --verbose
        with patch("sys.argv", ["run_eval.py", "--verbose"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", tmp_path / "runs"), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Debug logging messages appear
        captured = capsys.readouterr()
        assert "Verbose logging enabled" in captured.err or "DEBUG" in captured.err

    def test_cli_missing_golden_dataset_exits_gracefully(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI exits with helpful error when golden dataset not found.

        Verifies:
        - Missing dataset triggers FileNotFoundError
        - Error message is helpful and printed to stderr
        - Exit code is 1
        """
        # Arrange: Empty golden directory
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)

        # Act & Assert: CLI should exit with error
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.check_ollama_or_exit"), \
             pytest.raises(SystemExit) as exc_info:
            from scripts.run_eval import main
            main()

        # Assert: Error exit code
        assert exc_info.value.code == 1

        # Assert: Helpful error message in stderr
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert "No golden dataset found" in output or "not found" in output.lower()

    def test_cli_artifact_filename_follows_timestamp_format(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
    ) -> None:
        """Test artifact filename follows YYYY-MM-DDTHH-MM-SS.json pattern.

        Verifies:
        - Artifact filename has ISO timestamp format
        - Filename has correct structure with 'T' separator
        """
        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001])
        )

        # Arrange: Output directory
        output_dir = tmp_path / "runs"

        # Act: Run CLI
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", output_dir), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Filename follows timestamp pattern
        artifact_files = list(output_dir.glob("*.json"))
        artifact_path = artifact_files[0]
        assert "T" in artifact_path.stem, "Filename should have ISO timestamp with 'T' separator"
        assert artifact_path.stem.count("-") == 4, "Filename should have 4 dashes (YYYY-MM-DDTHH-MM-SS)"
        assert artifact_path.suffix == ".json"

    def test_cli_summary_output_format(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test summary table is printed with correct format.

        Verifies:
        - Dataset metadata is displayed
        - Overall scores are shown
        - Pass rate is formatted correctly
        - Next steps are printed
        """
        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=2)

        # Arrange: Mock chain with perfect matches
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.side_effect = [
            ChatResponse(**_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001, CHUNK_002])),
            ChatResponse(**_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001, CHUNK_002])),
        ]

        # Act: Run CLI
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", tmp_path / "runs"), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Summary components are present
        captured = capsys.readouterr()
        assert "EVALUATED DATASET:" in captured.out
        assert DATASET_SCOPE in captured.out
        assert "OVERALL SCORES" in captured.out
        assert "OVERALL PASS RATE" in captured.out
        assert "Evaluation complete" in captured.out
        assert "Next steps:" in captured.out

    def test_cli_production_defaults(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
    ) -> None:
        """Test CLI uses production defaults from src.configs.eval.

        Verifies:
        - DEFAULT_GOLDEN_DATASET_PATH is Path("evals/golden")
        - DEFAULT_EVAL_ARTIFACTS_PATH is Path("evals/runs")
        - These defaults are used when no CLI args provided
        """
        # Import and verify production constants
        from src.configs.eval import DEFAULT_GOLDEN_DATASET_PATH, DEFAULT_EVAL_ARTIFACTS_PATH
        assert DEFAULT_GOLDEN_DATASET_PATH == Path("evals/golden")
        assert DEFAULT_EVAL_ARTIFACTS_PATH == Path("evals/runs")

        # Arrange: Create golden dataset at production-relative path
        golden_dir = tmp_path / "evals" / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001, CHUNK_002])
        )

        # Act: Run with no arguments, using production defaults
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch("sys.argv", ["run_eval.py"]), \
                 patch("scripts.run_eval.build_chain", return_value=mock_chain), \
                 patch("scripts.run_eval.check_ollama_or_exit"):
                from scripts.run_eval import main
                main()
        finally:
            os.chdir(original_cwd)

        # Assert: Artifact written to production default path
        output_dir = tmp_path / "evals" / "runs"
        assert output_dir.exists(), "Should create directory at DEFAULT_EVAL_ARTIFACTS_PATH"
        artifact_files = list(output_dir.glob("*.json"))
        assert len(artifact_files) == 1, "Should save artifact to production default path"

        # Assert: Artifact is valid
        saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
        assert saved_run.dataset_scope == DATASET_SCOPE

    def test_cli_all_flags_combined(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI with all flags combined: custom paths + verbose.

        Verifies:
        - All flags work together correctly
        - Custom golden path is respected
        - Custom output path is used
        - Verbose logging is enabled
        """
        # Arrange: Create golden dataset at custom location
        custom_golden_dir = tmp_path / "my_custom_golden"
        custom_golden_dir.mkdir(parents=True)
        golden_path = _create_golden_dataset_file(
            directory=custom_golden_dir,
            version="10.0",
            num_examples=1,
        )

        # Arrange: Custom output location
        custom_output = tmp_path / "my_custom_output"

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001])
        )

        # Act: Run with all flags
        with patch("sys.argv", [
                "run_eval.py",
                "--golden-path", str(golden_path),
                "--output-path", str(custom_output),
                "--verbose",
            ]), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Artifact in custom output
        artifact_files = list(custom_output.glob("*.json"))
        assert len(artifact_files) == 1

        # Assert: Artifact reflects custom dataset
        saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
        assert saved_run.dataset_version == "10.0"

        # Assert: Verbose logging enabled
        captured = capsys.readouterr()
        assert "Verbose logging enabled" in captured.err or "DEBUG" in captured.err

    def test_cli_multi_author_dataset_succeeds(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI successfully processes dataset with multiple authors.

        Verifies:
        - Multi-author datasets are loaded correctly
        - Chain is built for each author
        - Eval runs successfully with multiple authors
        - Summary displays multiple authors
        """
        from src.configs.authors import AUTHOR_CONFIGS, AuthorConfig
        from tests.fake_authors import FAKE_AUTHOR_A, FAKE_AUTHOR_B

        # Arrange: Create fake author configs for validation
        # We need to patch AUTHOR_CONFIGS to include fake authors so that:
        # 1. GoldenExample validation passes when creating test data
        # 2. build_chain validation passes when building chains
        fake_author_configs = AUTHOR_CONFIGS.copy()
        fake_author_configs[FAKE_AUTHOR_A] = AuthorConfig(
            prompt_factory=lambda: Mock(),  # Not used in this test
            exit_message="Test exit message A",
        )
        fake_author_configs[FAKE_AUTHOR_B] = AuthorConfig(
            prompt_factory=lambda: Mock(),  # Not used in this test
            exit_message="Test exit message B",
        )

        # Arrange: Create multi-author golden dataset with mocked AUTHOR_CONFIGS
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)

        # Create dataset with multiple authors manually to control author list
        authors = [FAKE_AUTHOR_A, FAKE_AUTHOR_B]

        # Patch AUTHOR_CONFIGS during GoldenExample creation
        with patch("src.configs.authors.AUTHOR_CONFIGS", fake_author_configs):
            examples = [
                GoldenExample(
                    **_golden_example_kwargs(
                        id="example_0",
                        question=QUESTION_EN_001,
                        author=FAKE_AUTHOR_A,
                        language=ENGLISH_ISO_CODE,
                    )
                ),
                GoldenExample(
                    **_golden_example_kwargs(
                        id="example_1",
                        question=QUESTION_EN_002,
                        author=FAKE_AUTHOR_B,
                        language=ENGLISH_ISO_CODE,
                    )
                ),
            ]
            dataset = GoldenDataset(
                **_golden_dataset_kwargs(
                    authors=sorted(authors),
                    examples=examples,
                )
            )

        authors_str = "_".join(sorted(authors))
        golden_path = golden_dir / f"{DATASET_SCOPE}_{authors_str}_v{DATASET_VERSION}_{DATASET_DATE}.json"
        golden_path.write_text(dataset.model_dump_json(indent=2))

        # Arrange: Mock chains for both authors
        mock_chain_a = Mock(spec=Runnable)
        mock_chain_a.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001])
        )
        mock_chain_b = Mock(spec=Runnable)
        mock_chain_b.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_002])
        )

        def build_chain_for_author(author: str, **kwargs):
            if author == FAKE_AUTHOR_A:
                return mock_chain_a
            return mock_chain_b

        # Act: Run with explicit golden path (multi-author not auto-discovered)
        # Patch AUTHOR_CONFIGS again during main() execution for build_chain validation
        with patch("sys.argv", ["run_eval.py", "--golden-path", str(golden_path)]), \
             patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", tmp_path / "runs"), \
             patch("scripts.run_eval.build_chain", side_effect=build_chain_for_author), \
             patch("scripts.run_eval.check_ollama_or_exit"), \
             patch("src.configs.authors.AUTHOR_CONFIGS", fake_author_configs):
            from scripts.run_eval import main
            main()

        # Assert: Eval succeeds
        artifact_files = list((tmp_path / "runs").glob("*.json"))
        assert len(artifact_files) == 1

        # Assert: Artifact reflects multi-author dataset
        saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
        assert set(saved_run.dataset_authors) == {FAKE_AUTHOR_A, FAKE_AUTHOR_B}

        # Assert: Summary displays both authors
        captured = capsys.readouterr()
        assert FAKE_AUTHOR_A in captured.out or "condorcet" in captured.out
        assert FAKE_AUTHOR_B in captured.out or "wollstonecraft" in captured.out

    def test_cli_ollama_unavailable_exits_early(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI exits when Ollama unavailable.

        Verifies:
        - check_ollama_or_exit raises SystemExit
        - Error exit code is 1
        """
        # Arrange: Mock Ollama check to raise SystemExit
        def mock_check_ollama(logger):
            raise SystemExit(1)

        # Act & Assert: CLI should exit with error
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.check_ollama_or_exit", side_effect=mock_check_ollama), \
             pytest.raises(SystemExit) as exc_info:
            from scripts.run_eval import main
            main()

        # Assert: Error exit code
        assert exc_info.value.code == 1

    def test_cli_permission_error_exits_gracefully(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI exits gracefully on permission error for output directory.

        Verifies:
        - Read-only output directory triggers error
        - Error message is helpful
        - Exit code is 1
        """
        import os
        import stat

        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001])
        )

        # Arrange: Read-only output directory
        output_dir = tmp_path / "readonly_output"
        output_dir.mkdir(parents=True)
        os.chmod(output_dir, stat.S_IRUSR | stat.S_IXUSR)  # Read + execute only

        try:
            # Act & Assert: CLI should exit with error
            with patch("sys.argv", ["run_eval.py"]), \
                 patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
                 patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", output_dir), \
                 patch("scripts.run_eval.build_chain", return_value=mock_chain), \
                 patch("scripts.run_eval.check_ollama_or_exit"), \
                 pytest.raises(SystemExit) as exc_info:
                from scripts.run_eval import main
                main()

            # Assert: Error exit code
            assert exc_info.value.code == 1

            # Assert: Error message mentions permission
            captured = capsys.readouterr()
            output = captured.err + captured.out
            assert "permission" in output.lower() or "denied" in output.lower()

        finally:
            # Cleanup: Restore permissions
            os.chmod(output_dir, stat.S_IRWXU)

    def test_cli_chain_building_failure_exits_gracefully(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI exits gracefully when chain building fails.

        Verifies:
        - Exception during build_chain is caught
        - Error message is displayed
        - Exit code is 1
        """
        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock build_chain to raise exception
        def mock_build_chain_failing(**kwargs):
            raise RuntimeError("Failed to initialize chain")

        # Act & Assert: CLI should exit with error
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.build_chain", side_effect=mock_build_chain_failing), \
             patch("scripts.run_eval.check_ollama_or_exit"), \
             pytest.raises(SystemExit) as exc_info:
            from scripts.run_eval import main
            main()

        # Assert: Error exit code
        assert exc_info.value.code == 1

        # Assert: Error message displayed
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert "error" in output.lower()

    def test_cli_os_error_during_save_exits_gracefully(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI exits gracefully on OSError during save (e.g., disk full).

        Verifies:
        - OSError during save_eval_run is caught
        - Error message is helpful
        - Exit code is 1
        """
        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001])
        )

        # Arrange: Mock save_eval_run to raise OSError
        def mock_save_failing(eval_run, output_dir):
            raise OSError("No space left on device")

        # Act & Assert: CLI should exit with error
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"), \
             patch("scripts.run_eval.save_eval_run", side_effect=mock_save_failing), \
             pytest.raises(SystemExit) as exc_info:
            from scripts.run_eval import main
            main()

        # Assert: Error exit code
        assert exc_info.value.code == 1

        # Assert: Error message displayed
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert "no space" in output.lower() or "error" in output.lower()

    def test_cli_eval_run_failure_exits_gracefully(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test CLI exits gracefully when eval execution fails.

        Verifies:
        - Exception during run_eval is caught
        - Error message is displayed
        - Exit code is 1
        """
        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001])
        )

        # Arrange: Mock run_eval to raise exception
        def mock_run_eval_failing(golden_dataset, author_chains):
            raise RuntimeError("Evaluation failed")

        # Act & Assert: CLI should exit with error
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"), \
             patch("scripts.run_eval.run_eval", side_effect=mock_run_eval_failing), \
             pytest.raises(SystemExit) as exc_info:
            from scripts.run_eval import main
            main()

        # Assert: Error exit code
        assert exc_info.value.code == 1

        # Assert: Error message displayed
        captured = capsys.readouterr()
        output = captured.err + captured.out
        assert "error" in output.lower()

    def test_cli_error_messages_use_stderr(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test error messages are written to stderr, not stdout.

        Verifies:
        - Error messages appear in stderr
        - Stdout is not polluted with errors
        """
        # Arrange: Empty golden directory to trigger error
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)

        # Act: Run - should fail
        with patch("sys.argv", ["run_eval.py"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.check_ollama_or_exit"), \
             pytest.raises(SystemExit):
            from scripts.run_eval import main
            main()

        # Assert: Error appears in stderr
        captured = capsys.readouterr()
        assert "ERROR" in captured.err or "not found" in captured.err.lower()

    def test_cli_threshold_override_applies_to_all_metrics(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test --threshold flag overrides threshold for all metrics.

        Verifies:
        - --threshold flag is parsed correctly
        - Override applies to all registered metrics
        - effective_thresholds in EvalRun reflects the override
        """
        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001, CHUNK_002])
        )

        # Arrange: Output directory
        output_dir = tmp_path / "runs"

        # Act: Run with --threshold 0.95
        custom_threshold = 0.95
        with patch("sys.argv", ["run_eval.py", "--threshold", str(custom_threshold)]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", output_dir), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Artifact created
        artifact_files = list(output_dir.glob("*.json"))
        assert len(artifact_files) == 1

        # Assert: All metrics in effective_thresholds use the override value
        saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
        for metric_name, threshold in saved_run.effective_thresholds.items():
            assert threshold == custom_threshold, (
                f"Expected threshold {custom_threshold} for metric '{metric_name}', "
                f"got {threshold}"
            )

        # Assert: Log message confirms override
        captured = capsys.readouterr()
        assert f"Overriding all metric thresholds to: {custom_threshold}" in captured.err

    def test_cli_threshold_validation_rejects_out_of_range_values(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test --threshold flag rejects values outside [0.0, 1.0] range.

        Verifies:
        - Values < 0.0 are rejected
        - Values > 1.0 are rejected
        - Helpful error message is shown
        - Exit code is 1
        """
        # Test threshold > 1.0
        with patch("sys.argv", ["run_eval.py", "--threshold", "1.5"]), \
             pytest.raises(SystemExit) as exc_info:
            from scripts.run_eval import main
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "must be between 0.0 and 1.0" in captured.err
        assert "1.5" in captured.err

        # Test threshold < 0.0
        with patch("sys.argv", ["run_eval.py", "--threshold", "-0.1"]), \
             pytest.raises(SystemExit) as exc_info:
            from scripts.run_eval import main
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "must be between 0.0 and 1.0" in captured.err
        assert "-0.1" in captured.err

    def test_cli_threshold_boundary_values_are_accepted(
        self,
        tmp_path: Path,
        fake_chat_model: FakeChatModel,
    ) -> None:
        """Test --threshold accepts boundary values 0.0 and 1.0.

        Verifies:
        - --threshold 0.0 is valid
        - --threshold 1.0 is valid
        """
        # Arrange: Create golden dataset
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir(parents=True)
        _create_golden_dataset_file(directory=golden_dir, num_examples=1)

        # Arrange: Mock chain
        mock_chain = Mock(spec=Runnable)
        mock_chain.invoke.return_value = ChatResponse(
            **_chat_response_kwargs(retrieved_passage_ids=[CHUNK_001])
        )

        # Arrange: Output directory
        output_dir = tmp_path / "runs"

        # Test threshold = 0.0
        with patch("sys.argv", ["run_eval.py", "--threshold", "0.0"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", output_dir), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Artifact created with 0.0 thresholds
        artifact_files = list(output_dir.glob("*.json"))
        assert len(artifact_files) == 1
        saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
        for threshold in saved_run.effective_thresholds.values():
            assert threshold == 0.0

        # Clear output for next test
        for f in artifact_files:
            f.unlink()

        # Test threshold = 1.0
        with patch("sys.argv", ["run_eval.py", "--threshold", "1.0"]), \
             patch("scripts.run_eval.DEFAULT_GOLDEN_DATASET_PATH", golden_dir), \
             patch("scripts.run_eval.DEFAULT_EVAL_ARTIFACTS_PATH", output_dir), \
             patch("scripts.run_eval.build_chain", return_value=mock_chain), \
             patch("scripts.run_eval.check_ollama_or_exit"):
            from scripts.run_eval import main
            main()

        # Assert: Artifact created with 1.0 thresholds
        artifact_files = list(output_dir.glob("*.json"))
        assert len(artifact_files) == 1
        saved_run = EvalRun.model_validate_json(artifact_files[0].read_text())
        for threshold in saved_run.effective_thresholds.values():
            assert threshold == 1.0

