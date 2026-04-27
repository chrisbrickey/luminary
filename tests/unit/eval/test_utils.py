"""Unit tests for eval utility functions."""

import json
import os
import re
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tests.fake_authors import FAKE_AUTHOR_A, FAKE_AUTHOR_B, FAKE_AUTHOR_C
from src.configs.common import ENGLISH_ISO_CODE
from src.eval.utils import discover_latest_golden_dataset, load_eval_run, load_golden_dataset, save_eval_run, format_eval_report_stub
from src.schemas.eval import EvalRun, GoldenDataset

# --- Pytest fixtures ---


@pytest.fixture(autouse=True)
def _mock_authors(mock_author_configs):
    """Apply author mocking to all tests in this file."""
    # Also need to patch the eval.utils module since it imports DEFAULT_AUTHOR directly
    with patch("src.eval.utils.DEFAULT_AUTHOR", DEFAULT_TEST_AUTHOR):
        yield mock_author_configs


# --- Shared test constants ---

# Use centralized fake author constants from conftest
DEFAULT_TEST_AUTHOR = FAKE_AUTHOR_A
AUTHOR_A = FAKE_AUTHOR_A
AUTHOR_B = FAKE_AUTHOR_B
AUTHOR_C = FAKE_AUTHOR_C

# Golden dataset components
SCOPE = "persona"
VERSION_OLD = "1.0"
VERSION_NEW = "1.1"
DATE_OLD = "2025-06-12"
DATE_NEW = "2025-06-13"
NONEXISTENT_FILE = "does_not_exist.json"

# Test data constants
VALID_DATASET_JSON = {
    "scope": SCOPE,
    "authors": [AUTHOR_A],
    "version": VERSION_OLD,
    "created_date": DATE_OLD,
    "description": "Test dataset for unit tests",
    "examples": [
        {
            "id": "test_example_001",
            "question": "Sample question for testing?",
            "author": AUTHOR_A,
            "language": ENGLISH_ISO_CODE,
            "expected_chunk_ids": ["chunk_abc123"],
        }
    ],
}

INVALID_SCHEMA_JSON = {
    "version": VERSION_OLD,
    "created_date": DATE_OLD,
    # Missing required "description" field
    "examples": [],
}

INVALID_EVAL_RUN_SCHEMA_JSON = {
    "dataset_scope": SCOPE,
    # Missing all other required EvalRun fields
}

MALFORMED_JSON_CONTENT = '{"version": "1.0", "created_date": "2024-05-15"'  # Missing closing brace

# Test data for save_eval_run
VALID_EVAL_RUN_DICT = {
    "dataset_scope": SCOPE,
    "dataset_authors": [AUTHOR_A],
    "dataset_identifier": f"{SCOPE}_{AUTHOR_A}_v{VERSION_OLD}_{DATE_OLD}",
    "dataset_version": VERSION_OLD,
    "dataset_date": DATE_OLD,
    "run_timestamp": "2025-06-15T14:30:45+00:00",
    "system_version": {
        "chat_model": "test-model",
        "embedding_model": "test-embedding",
        "commit": "abc123def456",
        "k": "5",
        "chunk_size": "1000"
    },
    "effective_thresholds": {
        "test_metric": 0.7,
        "citation_quality": 0.80,
        "language_match": 0.95
    },
    "example_results": [
        {
            "example_id": "test_example_001",
            "question": "Sample question for testing?",
            "language": ENGLISH_ISO_CODE,
            "response": {
                "text": "Sample response",
                "retrieved_passage_ids": ["chunk_abc123"],
                "retrieved_contexts": ["Sample context text"],
                "retrieved_source_titles": ["Sample Source"],
                "language": ENGLISH_ISO_CODE
            },
            "metrics": [
                {
                    "name": "test_metric",
                    "score": 0.85,
                    "details": {"status": "ok"}
                }
            ],
            "passed": True
        }
    ],
    "aggregate_scores": {
        "overall": {
            "test_metric": 0.85,
            "citation_quality": 0.92,
            "language_match": 0.88
        }
    },
    "overall_pass_rate": 0.75
}


# --- Tests for load_golden_dataset() ---


def _golden_dataset_filename(scope: str, authors: list[str], version: str, date: str) -> str:
    """Generate a golden dataset filename following the naming convention.

    Pattern: {scope}_{authors}_v{version}_{YYYY-MM-DD}.json

    Args:
        scope: Dataset scope (e.g., "persona")
        authors: List of author names - will be sorted and joined with underscores
        version: Version string (e.g., "1.0")
        date: Date string in YYYY-MM-DD format
    """
    authors_str = "_".join(sorted(authors))
    return f"{scope}_{authors_str}_v{version}_{date}.json"


def test_load_golden_dataset_valid(tmp_path: Path) -> None:
    """Test loading a valid golden dataset JSON file."""
    dataset_file = tmp_path / "valid_dataset.json"
    dataset_file.write_text(json.dumps(VALID_DATASET_JSON))

    result = load_golden_dataset(dataset_file)

    assert isinstance(result, GoldenDataset)
    assert result.version == VERSION_OLD
    assert result.created_date == DATE_OLD
    assert result.description == "Test dataset for unit tests"
    assert len(result.examples) == 1
    assert result.examples[0].id == "test_example_001"


def test_load_golden_dataset_missing_file(tmp_path: Path) -> None:
    """Test that loading a nonexistent file raises FileNotFoundError with helpful message."""
    nonexistent_file = tmp_path / NONEXISTENT_FILE

    with pytest.raises(FileNotFoundError) as exc_info:
        load_golden_dataset(nonexistent_file)

    error_message = str(exc_info.value)
    assert NONEXISTENT_FILE in error_message
    assert "Golden dataset not found" in error_message


def test_load_golden_dataset_invalid_json(tmp_path: Path) -> None:
    """Test that loading malformed JSON raises JSONDecodeError."""
    malformed_file = tmp_path / "malformed.json"
    malformed_file.write_text(MALFORMED_JSON_CONTENT)

    with pytest.raises(json.JSONDecodeError):
        load_golden_dataset(malformed_file)


def test_load_golden_dataset_invalid_schema(tmp_path: Path) -> None:
    """Test that loading valid JSON with wrong schema raises ValidationError."""
    invalid_schema_file = tmp_path / "invalid_schema.json"
    invalid_schema_file.write_text(json.dumps(INVALID_SCHEMA_JSON))

    with pytest.raises(ValidationError):
        load_golden_dataset(invalid_schema_file)


def test_load_golden_dataset_permission_denied(tmp_path: Path) -> None:
    """Test that load_golden_dataset raises PermissionError with context when file is unreadable."""
    dataset_file = tmp_path / "unreadable.json"
    dataset_file.write_text(json.dumps(VALID_DATASET_JSON))

    # Make file unreadable (Unix-like systems only)
    os.chmod(dataset_file, 0o000)

    try:
        with pytest.raises(PermissionError) as exc_info:
            load_golden_dataset(dataset_file)

        error_message = str(exc_info.value)
        assert "Permission denied" in error_message
        assert str(dataset_file) in error_message
    finally:
        # Restore permissions for cleanup
        os.chmod(dataset_file, stat.S_IRUSR | stat.S_IWUSR)


def test_load_golden_dataset_is_directory_error(tmp_path: Path) -> None:
    """Test that load_golden_dataset raises IsADirectoryError when path is a directory."""
    # Create a directory instead of a file
    directory_path = tmp_path / "not_a_file"
    directory_path.mkdir()

    with pytest.raises(IsADirectoryError) as exc_info:
        load_golden_dataset(directory_path)

    error_message = str(exc_info.value)
    assert "Expected file but found directory" in error_message
    assert str(directory_path) in error_message


# --- Tests for discover_latest_golden_dataset() ---


def test_discover_latest_golden_dataset_finds_latest(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset returns the newest file by lexicographic sort."""
    # Create two dataset files with different versions and dates
    older_filename = _golden_dataset_filename(SCOPE, [AUTHOR_A], VERSION_OLD, DATE_OLD)
    newer_filename = _golden_dataset_filename(SCOPE, [AUTHOR_A], VERSION_NEW, DATE_NEW)

    older_file = tmp_path / older_filename
    newer_file = tmp_path / newer_filename

    older_file.write_text(json.dumps(VALID_DATASET_JSON))
    newer_file.write_text(json.dumps(VALID_DATASET_JSON))

    result = discover_latest_golden_dataset(tmp_path, authors=[AUTHOR_A])

    assert result == newer_file


def test_discover_latest_golden_dataset_lexicographic_sort_limitation(tmp_path: Path) -> None:
    """Test documenting limitation: multi-digit minor versions don't sort semantically.

    Known limitation: Lexicographic sorting means v1.9 > v1.10 (semantically incorrect).
    This is acceptable for the use case and because the schema limits minor versions to
    single-digit decimals. If this becomes a problem, implement semantic version parsing.
    """
    # Create files demonstrating the limitation
    v1_9_filename = _golden_dataset_filename(SCOPE, [AUTHOR_A], "1.9", DATE_OLD)
    v1_10_filename = _golden_dataset_filename(SCOPE, [AUTHOR_A], "1.10", DATE_OLD)

    v1_9_file = tmp_path / v1_9_filename
    v1_10_file = tmp_path / v1_10_filename

    v1_9_file.write_text(json.dumps(VALID_DATASET_JSON))
    v1_10_file.write_text(json.dumps(VALID_DATASET_JSON))

    result = discover_latest_golden_dataset(tmp_path, authors=[AUTHOR_A])

    # Documents current behavior: v1.9 sorts after v1.10 lexicographically
    # Lexicographic: "v1.9" > "v1.10" (because '9' > '1' at position 4)
    assert result == v1_9_file  # v1.9 comes first (semantically wrong but expected)
    assert result != v1_10_file


def test_discover_latest_golden_dataset_no_matches(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset raises FileNotFoundError when no files match."""
    expected_pattern = f"{SCOPE}_{AUTHOR_A}_v*.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        discover_latest_golden_dataset(tmp_path, authors=[AUTHOR_A])

    error_message = str(exc_info.value)
    assert expected_pattern in error_message
    assert str(tmp_path) in error_message


def test_discover_latest_golden_dataset_defaults_to_default_author(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset defaults to authors=[DEFAULT_AUTHOR] when not specified."""
    # Create a file with default author
    filename = _golden_dataset_filename(SCOPE, [DEFAULT_TEST_AUTHOR], VERSION_OLD, DATE_OLD)
    file_default_author = tmp_path / filename
    file_default_author.write_text(json.dumps(VALID_DATASET_JSON))

    # Call without specifying authors - should find the file with default author
    result = discover_latest_golden_dataset(tmp_path, scope=SCOPE)

    assert result == file_default_author
    assert DEFAULT_TEST_AUTHOR in result.name


def test_discover_latest_golden_dataset_respects_author(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset filters by authors parameter."""
    # Create dataset files for different authors
    author_a_filename = _golden_dataset_filename(SCOPE, [AUTHOR_A], VERSION_OLD, DATE_OLD)
    author_b_filename = _golden_dataset_filename(SCOPE, [AUTHOR_B], VERSION_OLD, DATE_OLD)

    author_a_file = tmp_path / author_a_filename
    author_b_file = tmp_path / author_b_filename

    author_a_file.write_text(json.dumps(VALID_DATASET_JSON))
    author_b_file.write_text(json.dumps(VALID_DATASET_JSON))

    result = discover_latest_golden_dataset(tmp_path, authors=[AUTHOR_A])

    assert result == author_a_file
    assert AUTHOR_A in result.name
    assert AUTHOR_B not in result.name


def test_discover_latest_golden_dataset_defaults_to_persona_scope(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset defaults to scope='persona' when not specified."""
    # Create a file with scope "persona"
    persona_filename = _golden_dataset_filename("persona", [AUTHOR_A], VERSION_OLD, DATE_OLD)
    persona_file = tmp_path / persona_filename
    persona_file.write_text(json.dumps(VALID_DATASET_JSON))

    # Call without specifying scope - should find the "persona" file
    result = discover_latest_golden_dataset(tmp_path, authors=[AUTHOR_A])

    assert result == persona_file
    assert "persona" in result.name


def test_discover_latest_golden_dataset_respects_scope_override(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset can override the default scope."""
    # Create files with different scopes
    persona_filename = _golden_dataset_filename("persona", [AUTHOR_A], VERSION_OLD, DATE_OLD)
    debate_filename = _golden_dataset_filename("debate", [AUTHOR_A], VERSION_OLD, DATE_OLD)

    persona_file = tmp_path / persona_filename
    debate_file = tmp_path / debate_filename

    persona_file.write_text(json.dumps(VALID_DATASET_JSON))
    debate_file.write_text(json.dumps(VALID_DATASET_JSON))

    # Call with explicit scope="debate" - should find the debate file, not persona
    result = discover_latest_golden_dataset(tmp_path, scope="debate", authors=[AUTHOR_A])

    assert result == debate_file
    assert "debate" in result.name
    assert "persona" not in result.name


def test_discover_latest_golden_dataset_multi_author_sorts_authors(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset returns the correct dataset,
    regardless of the order of the authors given.

    The filename convention requires authors to be sorted alphabetically.
    """
    # Create a dataset file with authors sorted alphabetically in filename
    # Helper function will sort: [AUTHOR_A, AUTHOR_B] → "author_a_author_b"
    multi_author_filename = _golden_dataset_filename(SCOPE, [AUTHOR_A, AUTHOR_B], VERSION_OLD, DATE_OLD)
    multi_author_file = tmp_path / multi_author_filename
    multi_author_file.write_text(json.dumps(VALID_DATASET_JSON))

    # Call with authors in REVERSE alphabetical order
    # Function should sort them before searching: [AUTHOR_B, AUTHOR_A] → "author_a_author_b"
    result = discover_latest_golden_dataset(tmp_path, authors=[AUTHOR_B, AUTHOR_A])

    # Should still find the file because function sorts author names before matching
    assert result == multi_author_file
    assert f"{AUTHOR_A}_{AUTHOR_B}" in result.name


# --- Tests for save_eval_run() ---


def test_save_eval_run_creates_directory(tmp_path: Path) -> None:
    """Test that save_eval_run creates the output directory if it doesn't exist."""
    nonexistent_dir = tmp_path / "new_directory" / "nested"
    eval_run = EvalRun(**VALID_EVAL_RUN_DICT)

    # Directory should not exist yet
    assert not nonexistent_dir.exists()

    # Save should create the directory
    result_path = save_eval_run(eval_run, nonexistent_dir)

    # Directory should now exist
    assert nonexistent_dir.exists()
    assert nonexistent_dir.is_dir()
    assert result_path.parent == nonexistent_dir


def test_save_eval_run_generates_valid_filename(tmp_path: Path) -> None:
    """Test that save_eval_run generates filename matching pattern {YYYY-MM-DD}T{HH-MM-SS}.json."""
    eval_run = EvalRun(**VALID_EVAL_RUN_DICT)

    result_path = save_eval_run(eval_run, tmp_path)

    # Filename should match pattern: {YYYY-MM-DD}T{HH-MM-SS}.json
    timestamp_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}"
    expected_pattern = rf"^{timestamp_pattern}\.json$"

    assert result_path.parent == tmp_path
    assert re.match(expected_pattern, result_path.name), (
        f"Filename '{result_path.name}' does not match expected pattern '{expected_pattern}'"
    )

    # Verify file was created and contains valid JSON
    assert result_path.exists()
    assert result_path.is_file()


def test_save_eval_run_preserves_all_fields(tmp_path: Path) -> None:
    """Test that save_eval_run preserves all EvalRun fields in JSON output."""
    eval_run = EvalRun(**VALID_EVAL_RUN_DICT)

    result_path = save_eval_run(eval_run, tmp_path)

    # Load saved JSON and reconstruct EvalRun
    saved_data = json.loads(result_path.read_text())
    loaded_eval_run = EvalRun(**saved_data)

    # Verify all top-level fields are preserved
    assert loaded_eval_run.dataset_scope == eval_run.dataset_scope
    assert loaded_eval_run.dataset_authors == eval_run.dataset_authors
    assert loaded_eval_run.dataset_identifier == eval_run.dataset_identifier
    assert loaded_eval_run.dataset_version == eval_run.dataset_version
    assert loaded_eval_run.dataset_date == eval_run.dataset_date
    assert loaded_eval_run.run_timestamp == eval_run.run_timestamp
    assert loaded_eval_run.system_version == eval_run.system_version
    assert loaded_eval_run.effective_thresholds == eval_run.effective_thresholds
    assert loaded_eval_run.aggregate_scores == eval_run.aggregate_scores
    assert loaded_eval_run.overall_pass_rate == eval_run.overall_pass_rate

    # Verify example_results structure is preserved
    assert len(loaded_eval_run.example_results) == len(eval_run.example_results)
    for loaded_ex, orig_ex in zip(loaded_eval_run.example_results, eval_run.example_results):
        assert loaded_ex.example_id == orig_ex.example_id
        assert loaded_ex.question == orig_ex.question
        assert loaded_ex.language == orig_ex.language
        assert loaded_ex.passed == orig_ex.passed
        assert loaded_ex.response.text == orig_ex.response.text
        assert loaded_ex.response.retrieved_passage_ids == orig_ex.response.retrieved_passage_ids
        assert len(loaded_ex.metrics) == len(orig_ex.metrics)


def test_save_eval_run_permission_error_on_directory_creation(tmp_path: Path) -> None:
    """Test that save_eval_run raises PermissionError or OSError when directory creation fails."""
    eval_run = EvalRun(**VALID_EVAL_RUN_DICT)

    # Create a file where we want to create a directory (will cause error)
    blocking_file = tmp_path / "blocking_file"
    blocking_file.write_text("content")

    # Try to create directory with same name as file (should fail)
    output_dir = blocking_file / "nested"

    with pytest.raises((PermissionError, OSError)) as exc_info:
        save_eval_run(eval_run, output_dir)

    error_message = str(exc_info.value)
    # Error message should contain helpful context
    assert str(output_dir) in error_message or str(blocking_file) in error_message


def test_save_eval_run_permission_error_on_file_write(tmp_path: Path) -> None:
    """Test that save_eval_run raises PermissionError when file write is denied."""
    eval_run = EvalRun(**VALID_EVAL_RUN_DICT)

    # Create read-only directory (Unix-like systems only)
    read_only_dir = tmp_path / "read_only"
    read_only_dir.mkdir()

    # Make directory read-only (no write permission)
    os.chmod(read_only_dir, stat.S_IRUSR | stat.S_IXUSR)

    try:
        with pytest.raises(PermissionError) as exc_info:
            save_eval_run(eval_run, read_only_dir)

        error_message = str(exc_info.value)
        assert "Permission denied" in error_message
        assert "write" in error_message.lower()
    finally:
        # Restore permissions for cleanup
        os.chmod(read_only_dir, stat.S_IRWXU)


# --- Tests for load_eval_run() ---


def test_load_eval_run_valid(tmp_path: Path) -> None:
    """Test loading a valid EvalRun JSON file."""
    artifact_file = tmp_path / "eval_run.json"
    artifact_file.write_text(json.dumps(VALID_EVAL_RUN_DICT))

    result = load_eval_run(artifact_file)

    assert isinstance(result, EvalRun)
    assert result.dataset_scope == SCOPE
    assert result.dataset_version == VERSION_OLD
    assert result.overall_pass_rate == VALID_EVAL_RUN_DICT["overall_pass_rate"]
    assert len(result.example_results) == 1
    assert result.example_results[0].example_id == "test_example_001"


def test_load_eval_run_missing_file(tmp_path: Path) -> None:
    """Test that loading a nonexistent file raises FileNotFoundError with helpful message."""
    nonexistent_file = tmp_path / NONEXISTENT_FILE

    with pytest.raises(FileNotFoundError) as exc_info:
        load_eval_run(nonexistent_file)

    error_message = str(exc_info.value)
    assert NONEXISTENT_FILE in error_message
    assert "Eval run artifact not found" in error_message


def test_load_eval_run_invalid_json(tmp_path: Path) -> None:
    """Test that loading malformed JSON raises JSONDecodeError."""
    malformed_file = tmp_path / "malformed_run.json"
    malformed_file.write_text(MALFORMED_JSON_CONTENT)

    with pytest.raises(json.JSONDecodeError):
        load_eval_run(malformed_file)


def test_load_eval_run_invalid_schema(tmp_path: Path) -> None:
    """Test that loading valid JSON with wrong schema raises ValidationError."""
    invalid_schema_file = tmp_path / "invalid_run_schema.json"
    invalid_schema_file.write_text(json.dumps(INVALID_EVAL_RUN_SCHEMA_JSON))

    with pytest.raises(ValidationError):
        load_eval_run(invalid_schema_file)


def test_load_eval_run_permission_denied(tmp_path: Path) -> None:
    """Test that load_eval_run raises PermissionError with context when file is unreadable."""
    artifact_file = tmp_path / "unreadable_run.json"
    artifact_file.write_text(json.dumps(VALID_EVAL_RUN_DICT))

    os.chmod(artifact_file, 0o000)

    try:
        with pytest.raises(PermissionError) as exc_info:
            load_eval_run(artifact_file)

        error_message = str(exc_info.value)
        assert "Permission denied" in error_message
        assert str(artifact_file) in error_message
    finally:
        os.chmod(artifact_file, stat.S_IRUSR | stat.S_IWUSR)


def test_load_eval_run_is_directory_error(tmp_path: Path) -> None:
    """Test that load_eval_run raises IsADirectoryError when path is a directory."""
    directory_path = tmp_path / "not_a_run_file"
    directory_path.mkdir()

    with pytest.raises(IsADirectoryError) as exc_info:
        load_eval_run(directory_path)

    error_message = str(exc_info.value)
    assert "Expected file but found directory" in error_message
    assert str(directory_path) in error_message


# --- Tests for format_eval_report_stub() ---


def test_format_eval_report_stub_includes_all_sections(tmp_path: Path) -> None:
    """Test that format_eval_report_stub produces markdown including section headers.

    This test uses a list of required sections instead of parsing the actual
    template file because the real file is covered by integration tests."""
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(VALID_EVAL_RUN_DICT))

    result = format_eval_report_stub(artifact_path)

    assert "# Eval Report" in result
    assert "## Source Data" in result
    assert "## System Version" in result
    assert "## Eval Run Summary" in result
    assert "## Issue Analysis" in result
    assert "## Changes Made" in result
    assert "## Changes Deferred" in result


def test_format_eval_report_stub_includes_metadata(tmp_path: Path) -> None:
    """Test that format_eval_report_stub pre-populates metadata into the output markdown."""
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(VALID_EVAL_RUN_DICT))
    eval_run = EvalRun(**VALID_EVAL_RUN_DICT)

    result = format_eval_report_stub(artifact_path)

    # Verify title contains timestamp in YYYY-MM-DDTHH-MM-SS format
    timestamp_pattern = r"# Eval Report \d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}"
    assert re.search(timestamp_pattern, result), "Title should contain timestamp in YYYY-MM-DDTHH-MM-SS format"

    # Verify Source Data section contains eval run artifact path and dataset identifier
    assert str(artifact_path) in result, "Should include eval artifact path"
    assert eval_run.dataset_identifier in result, f"Should include dataset identifier: {eval_run.dataset_identifier}"

    # Verify System Version section contains all required fields
    assert eval_run.system_version["chat_model"] in result, "Should include chat_model"
    assert eval_run.system_version["embedding_model"] in result, "Should include embedding_model"
    assert f"k={eval_run.system_version['k']}" in result, "Should include retrieval config k"
    assert f"chunk_size={eval_run.system_version['chunk_size']}" in result, "Should include retrieval config chunk_size"


def test_format_eval_report_stub_includes_metrics_table(tmp_path: Path) -> None:
    """Test that format_eval_report_stub includes complete metrics table."""
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(VALID_EVAL_RUN_DICT))
    eval_run = EvalRun(**VALID_EVAL_RUN_DICT)

    result = format_eval_report_stub(artifact_path)

    # Verify table header exists
    assert "| Metric Name | Effective Threshold | Score | Status |" in result, "Should include metrics table header"

    # Verify table contains rows for all metrics
    for metric_name, score in eval_run.aggregate_scores["overall"].items():
        assert metric_name in result, f"Should include metric: {metric_name}"

        threshold = eval_run.effective_thresholds[metric_name]
        assert f"{threshold:.2f}" in result or f"{threshold}" in result, f"Should include threshold for {metric_name}"
        assert f"{score:.2f}" in result or f"{score}" in result, f"Should include score for {metric_name}"

        assert "Pass" in result or "Fail" in result, "Should include Pass/Fail status"

    # Verify overall pass rate is included and formatted as percentage
    expected_pass_rate = VALID_EVAL_RUN_DICT["overall_pass_rate"] * 100
    assert f"{expected_pass_rate:.1f}%" in result, "Should include overall pass rate as percentage"


def test_format_eval_report_stub_raises_on_missing_artifact(tmp_path: Path) -> None:
    """Test that format_eval_report_stub raises FileNotFoundError when artifact doesn't exist."""
    nonexistent_artifact = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        format_eval_report_stub(nonexistent_artifact)

    error_message = str(exc_info.value)
    assert "nonexistent.json" in error_message, "Error message should include the missing path"


def test_format_eval_report_stub_raises_on_missing_template(tmp_path: Path) -> None:
    """Test that format_eval_report_stub raises FileNotFoundError when template doesn't exist."""
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(VALID_EVAL_RUN_DICT))

    with patch("src.eval.utils.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        with pytest.raises(FileNotFoundError) as exc_info:
            format_eval_report_stub(artifact_path)

        error_message = str(exc_info.value)
        assert "template" in error_message.lower(), "Error message should mention template"



