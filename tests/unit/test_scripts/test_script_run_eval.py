"""Unit tests for scripts/run_eval.py.

This file contains unit tests for the print_summary_table() public method,
where the EvalRun is mocked.

All other functionality in scripts/run_eval.py is tested through the main()
method in tests/integration/test_eval_cli_integration.py, as it involves
CLI argument parsing, file I/O, and cross-module orchestration.
"""


import pytest

from src.configs.common import ENGLISH_ISO_CODE, FRENCH_ISO_CODE
from tests.fake_authors import FAKE_AUTHOR_A, FAKE_AUTHOR_B
from src.schemas.eval import EvalRun, ExampleResult, MetricResult
from src.schemas.chat import ChatResponse
from scripts.run_eval import print_summary_table

# Test constants
TEST_QUESTION = "What is progress?"
TEST_RESPONSE_TEXT = "Progress is the advancement of human knowledge and reason."
TEST_CHUNK_IDS = ["abc123def456", "xyz789ghi012"]
TEST_CONTEXTS = ["Context about progress.", "Context about reason."]
TEST_SOURCE_TITLES = ["Essay on Progress, Page 5"]

METRIC_NAME = "retrieval_relevance"
SCORE = 0.85
THRESHOLD = 0.8

CHAT_MODEL = "llama3"
COMMIT = "def456"
EVAL_RUN_TIMESTAMP = "2026-04-01T12:00:00Z"

DATASET_SCOPE = "persona"
DATASET_AUTHORS = [FAKE_AUTHOR_A]
DATASET_VERSION = "2.0"
DATASET_DATE = "2025-02-01"
DATASET_IDENTIFIER = f"{DATASET_SCOPE}_{"_".join(DATASET_AUTHORS)}_v{DATASET_VERSION}_{DATASET_DATE}"


def create_mock_eval_run(
    dataset_identifier: str = DATASET_IDENTIFIER,
    dataset_scope: str = DATASET_SCOPE,
    dataset_authors: list[str] | None = None,
    dataset_version: str = DATASET_VERSION,
    dataset_date: str = DATASET_DATE,
    overall_pass_rate: float = 0.75,
    aggregate_scores: dict | None = None,
    effective_thresholds: dict | None = None,
    num_examples: int = 1,
) -> EvalRun:
    """Create a mock EvalRun for testing print_summary_table.
    Returns: EvalRun instance"""

    # Setup defaults to prevent mutable default parameters
    if dataset_authors is None:
        dataset_authors = DATASET_AUTHORS

    if aggregate_scores is None:
        aggregate_scores = {"overall": {METRIC_NAME: SCORE}}

    if effective_thresholds is None:
        effective_thresholds = {METRIC_NAME: THRESHOLD}

    # Create example results
    example_results = []
    for i in range(num_examples):
        example_results.append(
            ExampleResult(
                example_id=f"example_{i}",
                question=TEST_QUESTION,
                language=ENGLISH_ISO_CODE,
                response=ChatResponse(
                    text=TEST_RESPONSE_TEXT,
                    retrieved_passage_ids=TEST_CHUNK_IDS,
                    retrieved_contexts=TEST_CONTEXTS,
                    retrieved_source_titles=TEST_SOURCE_TITLES,
                    language=ENGLISH_ISO_CODE,
                ),
                metrics=[
                    MetricResult(name=METRIC_NAME, score=SCORE, details={}),
                ],
                passed=True,
            )
        )

    return EvalRun(
        dataset_scope=dataset_scope,
        dataset_authors=dataset_authors,
        dataset_identifier=dataset_identifier,
        dataset_version=dataset_version,
        dataset_date=dataset_date,
        run_timestamp=EVAL_RUN_TIMESTAMP,
        system_version={"chat_model": CHAT_MODEL, "commit": COMMIT, "timestamp": EVAL_RUN_TIMESTAMP},
        effective_thresholds=effective_thresholds,
        example_results=example_results,
        aggregate_scores=aggregate_scores,
        overall_pass_rate=overall_pass_rate,
    )


class TestPrintSummaryTable:
    """Test print_summary_table() function."""

    def test_prints_dataset_metadata(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that dataset metadata is displayed correctly."""
        eval_run = create_mock_eval_run()

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert DATASET_IDENTIFIER in captured.out
        assert f"Scope: {DATASET_SCOPE}" in captured.out
        assert f"Authors: {"_".join(DATASET_AUTHORS)}" in captured.out
        assert f"Version: {DATASET_VERSION}" in captured.out
        assert f"Date: {DATASET_DATE}" in captured.out

    def test_prints_other_metadata_and_symbols(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that system version information is displayed."""
        eval_run = create_mock_eval_run()

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert f"Chat Model: {CHAT_MODEL}" in captured.out
        assert f"Git Commit: {COMMIT}" in captured.out
        assert f"EVAL RUN TIMESTAMP: {EVAL_RUN_TIMESTAMP}" in captured.out
        assert "✅" in captured.out # pass symbol when score >= threshold

        # Verify that language breakdown and cross-langugage metrics not displayed when not present in the input
        assert f"ONLY" not in captured.out
        assert "CROSS-LANGUAGE METRICS" not in captured.out

    def test_prints_fail_symbol_when_score_below_threshold(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that ❌ appears when score is below threshold."""
        eval_run = create_mock_eval_run(
            aggregate_scores={"overall": {METRIC_NAME: 0.65}},
            effective_thresholds={METRIC_NAME: THRESHOLD},
        )

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert "❌" in captured.out

    def test_prints_by_language_breakdown_when_present(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that by-language breakdown is displayed when available."""
        french_score, english_score = 0.88, 0.82
        eval_run = create_mock_eval_run(
            aggregate_scores={
                "overall": {METRIC_NAME: SCORE},
                "by_language": {
                    FRENCH_ISO_CODE: {METRIC_NAME: french_score},
                    ENGLISH_ISO_CODE : {METRIC_NAME: english_score},
                },
            },
            effective_thresholds={METRIC_NAME: THRESHOLD},
        )

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert f"{FRENCH_ISO_CODE.upper()} ONLY" in captured.out
        assert f"{ENGLISH_ISO_CODE.upper()} ONLY" in captured.out
        assert f"{french_score}" in captured.out
        assert f"{english_score}" in captured.out

    def test_prints_cross_language_metrics_when_present(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that cross-language metrics section appears when available."""
        name2, score2, threshold2,  = "translation_consistency", 0.78, 0.7
        eval_run = create_mock_eval_run(
            aggregate_scores={
                "overall": {METRIC_NAME: SCORE},
                "cross_language": {name2 : score2},
            },
            effective_thresholds={ METRIC_NAME: THRESHOLD, name2: threshold2 },
        )

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert "CROSS-LANGUAGE METRICS" in captured.out
        assert f"{name2}" in captured.out
        assert f"{score2}" in captured.out

    def test_prints_overall_scores_with_thresholds(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that overall scores are displayed with their thresholds."""
        name2, score2, threshold2,  = "answer_correctness", 0.72, 0.7
        eval_run = create_mock_eval_run(
            aggregate_scores={
                "overall": { METRIC_NAME: SCORE, name2: score2 }
            },
            effective_thresholds={ METRIC_NAME: THRESHOLD, name2: threshold2,
            },
        )

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert "OVERALL SCORES" in captured.out
        assert METRIC_NAME in captured.out
        assert f"{SCORE}" in captured.out
        assert f"threshold: {THRESHOLD}" in captured.out
        assert f"{name2}" in captured.out
        assert f"{score2}" in captured.out
        assert f"threshold: {threshold2}" in captured.out

    def test_prints_overall_pass_rate_with_fraction(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that overall pass rate displays as percentage with fraction."""
        pass_rate, example_count = 0.75, 4
        eval_run = create_mock_eval_run(
            overall_pass_rate=pass_rate,
            num_examples=example_count,
        )
        # Mark 3 out of 4 as passed
        eval_run.example_results[0].passed = True
        eval_run.example_results[1].passed = True
        eval_run.example_results[2].passed = True
        eval_run.example_results[3].passed = False

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert "OVERALL PASS RATE" in captured.out
        assert f"{pass_rate:.1%}" in captured.out
        assert f"(3/{example_count} examples)" in captured.out

    def test_perfect_pass_rate_displays_correctly(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that 100% pass rate is displayed correctly."""
        pass_rate, example_count = 1.0, 3
        eval_run = create_mock_eval_run(overall_pass_rate=pass_rate, num_examples=example_count)
        eval_run.example_results[0].passed = True
        eval_run.example_results[1].passed = True
        eval_run.example_results[2].passed = True

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert f"{pass_rate:.1%}" in captured.out
        assert f"({example_count}/{example_count} examples)" in captured.out

    def test_zero_pass_rate_displays_correctly(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that 0% pass rate is displayed correctly."""
        pass_rate, example_count = 0.0, 2
        eval_run = create_mock_eval_run(overall_pass_rate=pass_rate, num_examples=example_count)
        eval_run.example_results[0].passed = False
        eval_run.example_results[1].passed = False

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert f"{pass_rate:.1%}" in captured.out
        assert f"(0/{example_count} examples)" in captured.out

    def test_handles_multi_author_dataset(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that multiple authors are displayed correctly."""
        dataset_authors = [FAKE_AUTHOR_A, FAKE_AUTHOR_B]
        identifier = f"{DATASET_SCOPE}_{"_".join(dataset_authors)}_v{DATASET_VERSION}_{DATASET_DATE}"
        eval_run = create_mock_eval_run(dataset_identifier=identifier, dataset_authors=dataset_authors)

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert f"Authors: {", ".join(dataset_authors)}" in captured.out

    def test_formats_multiple_metrics_sorted_alphabetically(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that multiple metrics are displayed in alphabetical order."""
        a_metric, m_metric, z_metric = "a_metric", "m_metric", "z_metric"
        eval_run = create_mock_eval_run(
            aggregate_scores={ "overall": { z_metric: 0.90, a_metric: SCORE, m_metric: 0.78 } },
            effective_thresholds={ z_metric: THRESHOLD, a_metric: THRESHOLD, m_metric: THRESHOLD },
        )

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        output = captured.out

        # Verify alphabetical order
        a_pos = output.find(a_metric)
        m_pos = output.find(m_metric)
        z_pos = output.find(z_metric)
        assert a_pos < m_pos < z_pos, "Metrics should be sorted alphabetically"

    def test_handles_empty_metrics_gracefully(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that empty overall metrics dict shows 'No metrics available'."""
        eval_run = create_mock_eval_run(
            aggregate_scores={"overall": {}},
        )

        print_summary_table(eval_run)

        captured = capsys.readouterr()
        assert "No metrics available" in captured.out
