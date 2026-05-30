"""Integration tests for eval report stub formatting and CLI.

Tests the real component wiring:
- Real format_eval_report_stub function
- Real TEMPLATE.md file on disk
- Real EvalRun serialization / deserialization (save_eval_run + load_eval_run)
- Real SystemSnapshot field iteration
- Real CLI script (scripts/stub_eval_report.py) end-to-end with real file I/O
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.configs.authors import DEFAULT_AUTHOR
from src.configs.common import ENGLISH_ISO_CODE
from src.eval.utils import format_eval_report_stub, save_eval_run
from src.schemas.chat import ChatResponse
from src.schemas.eval import (
    AggregateScores,
    EvalRun,
    ExampleResult,
    GoldenDataset,
    GoldenExample,
    MetricResult,
    SystemSnapshot,
)


# --- Test constants ---

DATASET_SCOPE = "persona"
DATASET_VERSION = "3.0"
DATASET_DATE = "2029-01-15"
DATASET_DESCRIPTION = "test dataset for stub integration testing"

EXAMPLE_ID = "example_en_001"
QUESTION = "What is tolerance?"
CHUNK_001 = "chunk_001"
CONTEXT_001 = "Context about tolerance."
SOURCE_TITLE_001 = "Test Title, Page 12"

EXAMPLE_ID_2 = "example_en_002"
QUESTION_2 = "What is free speech?"
CHUNK_002 = "chunk_002"
CONTEXT_002 = "Context about free speech."
SOURCE_TITLE_002 = "Test Title, Page 34"

METRIC_NAME = "retrieval_recall"
METRIC_SCORE = 0.75       # above threshold → Pass
METRIC_SCORE_FAIL = 0.40  # below threshold → Fail
METRIC_THRESHOLD = 0.60
OVERALL_PASS_RATE = 0.5   # 1 of 2 examples passes
OVERALL_AVERAGE = 0.575   # (METRIC_SCORE + METRIC_SCORE_FAIL) / 2

RUN_TIMESTAMP = "2029-01-15T08:00:00+00:00"
STUB_CREATED_AT = "2029-01-16T09:00:00"  # ISO 8601, format_eval_report_stub calls format_timestamp internally
SNAPSHOT_COMMIT = "abc1234"
SNAPSHOT_CHAT_MODEL = "test-chat-model"
SNAPSHOT_EMBEDDING_MODEL = "test-embedding-model"
SNAPSHOT_CHUNK_COUNT = "4"
SNAPSHOT_CHUNK_SIZE = "512"


# --- Helper factories ---


def _make_eval_run() -> EvalRun:
    """Build a realistic EvalRun with mixed pass/fail results for stub formatting tests."""
    examples = [
        GoldenExample(
            id=EXAMPLE_ID,
            question=QUESTION,
            author=DEFAULT_AUTHOR,
            language=ENGLISH_ISO_CODE,
            expected_chunk_ids=[CHUNK_001],
        ),
        GoldenExample(
            id=EXAMPLE_ID_2,
            question=QUESTION_2,
            author=DEFAULT_AUTHOR,
            language=ENGLISH_ISO_CODE,
            expected_chunk_ids=[CHUNK_002],
        ),
    ]
    dataset = GoldenDataset(
        scope=DATASET_SCOPE,
        authors=[DEFAULT_AUTHOR],
        version=DATASET_VERSION,
        created_date=DATASET_DATE,
        description=DATASET_DESCRIPTION,
        examples=examples,
    )
    passing_result = ExampleResult(
        example_id=EXAMPLE_ID,
        question=QUESTION,
        language=ENGLISH_ISO_CODE,
        response=ChatResponse(
            text="Sample response.",
            retrieved_passage_ids=[CHUNK_001],
            retrieved_contexts=[CONTEXT_001],
            retrieved_source_titles=[SOURCE_TITLE_001],
            language=ENGLISH_ISO_CODE,
        ),
        metrics=[MetricResult(name=METRIC_NAME, score=METRIC_SCORE)],
        passed=True,
    )
    failing_result = ExampleResult(
        example_id=EXAMPLE_ID_2,
        question=QUESTION_2,
        language=ENGLISH_ISO_CODE,
        response=ChatResponse(
            text="Another sample response.",
            retrieved_passage_ids=[CHUNK_002],
            retrieved_contexts=[CONTEXT_002],
            retrieved_source_titles=[SOURCE_TITLE_002],
            language=ENGLISH_ISO_CODE,
        ),
        metrics=[MetricResult(name=METRIC_NAME, score=METRIC_SCORE_FAIL)],
        passed=False,
    )
    return EvalRun(
        run_timestamp=RUN_TIMESTAMP,
        golden_dataset=dataset,
        system_snapshot=SystemSnapshot(
            commit=SNAPSHOT_COMMIT,
            chat_model=SNAPSHOT_CHAT_MODEL,
            embedding_model=SNAPSHOT_EMBEDDING_MODEL,
            retrieval_chunk_count=SNAPSHOT_CHUNK_COUNT,
            retrieval_chunk_size=SNAPSHOT_CHUNK_SIZE,
        ),
        effective_thresholds={METRIC_NAME: METRIC_THRESHOLD},
        overall_pass_rate=OVERALL_PASS_RATE,
        overall_average=OVERALL_AVERAGE,
        aggregate_scores=AggregateScores(
            averages_by_metric={METRIC_NAME: OVERALL_AVERAGE},
            averages_by_language_and_metric={ENGLISH_ISO_CODE: {METRIC_NAME: OVERALL_AVERAGE}},
        ),
        example_results=[passing_result, failing_result],
    )


def _save_artifact(tmp_path: Path) -> Path:
    """Save a minimal EvalRun to tmp_path and return the artifact path."""
    return save_eval_run(_make_eval_run(), tmp_path)


# --- Integration tests ---


class TestFormatEvalReportStub:
    """Integration tests for constructing the pre-populated markdown of an evaluation report

    These tests verify that real TEMPLATE.md fields are populated correctly
    from a real EvalRun artifact using real file I/O.
    """

    def test_returns_markdown_with_report_header(self, tmp_path: Path) -> None:
        """Test that the return value is a non-empty string with the report header."""
        artifact_path = _save_artifact(tmp_path)

        result = format_eval_report_stub(artifact_path, STUB_CREATED_AT)

        assert isinstance(result, str)
        assert "# Eval Report" in result

    def test_populates_source_data_section(self, tmp_path: Path) -> None:
        """Test that artifact path and dataset identifier appear in Source Data."""
        artifact_path = _save_artifact(tmp_path)
        expected_identifier = f"{DATASET_SCOPE}_{DEFAULT_AUTHOR}_v{DATASET_VERSION}_{DATASET_DATE}"

        result = format_eval_report_stub(artifact_path, STUB_CREATED_AT)

        assert str(artifact_path) in result
        assert expected_identifier in result

    def test_populates_all_system_snapshot_fields(self, tmp_path: Path) -> None:
        """Test that every non-None SystemSnapshot value appears in the output.

        Iterates model_fields so new fields are automatically covered without
        manual test updates.
        """
        eval_run = _make_eval_run()
        artifact_path = save_eval_run(eval_run, tmp_path)

        result = format_eval_report_stub(artifact_path, STUB_CREATED_AT)

        for field_name, field_info in SystemSnapshot.model_fields.items():
            value = getattr(eval_run.system_snapshot, field_name)
            if value is not None and field_info.title is not None:
                assert value in result, (
                    f"Expected system_snapshot.{field_name} value '{value}' in output"
                )

    def test_populates_metrics_table_with_name_threshold_score_and_status(self, tmp_path: Path) -> None:
        """Test that metric name, threshold, aggregate score, and Fail status appear in output.

        The table renders one row per metric using the aggregate (average) score, not individual
        example scores. OVERALL_AVERAGE (0.575) < METRIC_THRESHOLD (0.60), so status is Fail.
        """
        artifact_path = _save_artifact(tmp_path)

        result = format_eval_report_stub(artifact_path, STUB_CREATED_AT)

        assert METRIC_NAME in result
        assert f"{METRIC_THRESHOLD:.2f}" in result
        assert "Fail" in result

    def test_populates_overall_scores(self, tmp_path: Path) -> None:
        """Test that overall_pass_rate and overall_average are formatted and present in output."""
        artifact_path = _save_artifact(tmp_path)

        result = format_eval_report_stub(artifact_path, STUB_CREATED_AT)

        assert f"{OVERALL_PASS_RATE * 100:.1f}%" in result
        assert f"{OVERALL_AVERAGE:.2f}" in result

    def test_raises_file_not_found_for_missing_artifact(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when artifact does not exist."""
        missing_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            format_eval_report_stub(missing_path, STUB_CREATED_AT)


class TestStubEvalReportCLI:
    """Integration tests for the CLI script that manages the creation of a pre-populated eval report.

    These tests verify the full CLI workflow with real file I/O:
    - Real artifact read from disk
    - Real format_eval_report_stub (not mocked)
    - Real TEMPLATE.md on disk
    - Output file written to tmp_path
    """

    def test_cli_creates_report_file_with_populated_content(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test end-to-end: real artifact → real formatting → report file on disk.

        Verifies:
        - Output file is created with the expected filename pattern
        - Content has source data, system snapshot, metrics, and pass rate populated
        - Success message is printed to stdout
        """
        # Arrange: save a real artifact to a subdirectory (avoids glob collision with output)
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        artifact_path = _save_artifact(artifact_dir)
        output_dir = tmp_path / "reports"

        # Act: run CLI with real format_eval_report_stub
        with patch("sys.argv", ["stub_eval_report.py", str(artifact_path), "--output-path", str(output_dir)]):
            from scripts.stub_eval_report import main
            main()

        # Assert: exactly one report file created with correct filename pattern
        report_files = list(output_dir.glob("eval_report_*.md"))
        assert len(report_files) == 1
        assert re.match(r"^eval_report_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.md$", report_files[0].name)

        # Assert: content has all key populated fields
        content = report_files[0].read_text()
        expected_identifier = f"{DATASET_SCOPE}_{DEFAULT_AUTHOR}_v{DATASET_VERSION}_{DATASET_DATE}"
        assert str(artifact_path) in content
        assert expected_identifier in content
        assert SNAPSHOT_COMMIT in content
        assert METRIC_NAME in content
        assert f"{OVERALL_PASS_RATE * 100:.1f}%" in content
        assert f"{OVERALL_AVERAGE:.2f}" in content

        # Assert: success message printed to stdout
        captured = capsys.readouterr()
        assert report_files[0].name in captured.out

    def test_cli_exits_with_error_for_missing_artifact(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that CLI exits with code 1 and prints to stderr when artifact is missing."""
        missing_path = tmp_path / "nonexistent.json"

        with patch("sys.argv", ["stub_eval_report.py", str(missing_path)]), \
             pytest.raises(SystemExit) as exc_info:
            from scripts.stub_eval_report import main
            main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert str(missing_path) in captured.err
