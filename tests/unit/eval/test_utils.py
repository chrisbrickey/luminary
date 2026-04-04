"""Unit tests for eval utility functions."""

import json
import os
import re
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.utils import discover_latest_golden_dataset, load_golden_dataset, save_eval_run
from src.schemas.chat import ChatResponse
from src.schemas.eval import EvalRun, ExampleResult, GoldenDataset, MetricResult

# --- Shared test constants ---

# Generic author identifiers (not actual philosopher names)
AUTHOR_A = "author_a"
AUTHOR_B = "author_b"

# File naming components
DATASET_SCOPE = "golden"
VERSION_OLD = "1.0"
VERSION_NEW = "1.1"
DATE_OLD = "2025-06-12"
DATE_NEW = "2025-06-13"

# Common filenames
NONEXISTENT_FILE = "does_not_exist.json"

# Test data constants
VALID_DATASET_JSON = {
    "name": f"{AUTHOR_A}_golden",
    "version": VERSION_OLD,
    "created_date": DATE_OLD,
    "description": "Test dataset for unit tests",
    "examples": [
        {
            "id": "test_example_001",
            "question": "Sample question for testing?",
            "language": "en",
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

MALFORMED_JSON_CONTENT = '{"version": "1.0", "created_date": "2024-05-15"'  # Missing closing brace


def _golden_dataset_filename(author: str, version: str, date: str) -> str:
    """Generate a golden dataset filename following the naming convention.

    Pattern: {author}_{scope}_v{version}_{YYYY-MM-DD}.json
    """
    return f"{author}_{DATASET_SCOPE}_v{version}_{date}.json"


# --- Tests for load_golden_dataset() ---


def test_load_valid_dataset(tmp_path: Path) -> None:
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


def test_load_missing_file(tmp_path: Path) -> None:
    """Test that loading a nonexistent file raises FileNotFoundError with helpful message."""
    nonexistent_file = tmp_path / NONEXISTENT_FILE

    with pytest.raises(FileNotFoundError) as exc_info:
        load_golden_dataset(nonexistent_file)

    error_message = str(exc_info.value)
    assert NONEXISTENT_FILE in error_message
    assert "Golden dataset not found" in error_message


def test_load_invalid_json(tmp_path: Path) -> None:
    """Test that loading malformed JSON raises JSONDecodeError."""
    malformed_file = tmp_path / "malformed.json"
    malformed_file.write_text(MALFORMED_JSON_CONTENT)

    with pytest.raises(json.JSONDecodeError):
        load_golden_dataset(malformed_file)


def test_load_invalid_schema(tmp_path: Path) -> None:
    """Test that loading valid JSON with wrong schema raises ValidationError."""
    invalid_schema_file = tmp_path / "invalid_schema.json"
    invalid_schema_file.write_text(json.dumps(INVALID_SCHEMA_JSON))

    with pytest.raises(ValidationError):
        load_golden_dataset(invalid_schema_file)


def test_load_permission_denied(tmp_path: Path) -> None:
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


def test_load_is_directory_error(tmp_path: Path) -> None:
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


def test_discover_finds_latest(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset returns the newest file by lexicographic sort."""
    # Create two dataset files with different versions and dates
    older_filename = _golden_dataset_filename(AUTHOR_A, VERSION_OLD, DATE_OLD)
    newer_filename = _golden_dataset_filename(AUTHOR_A, VERSION_NEW, DATE_NEW)

    older_file = tmp_path / older_filename
    newer_file = tmp_path / newer_filename

    older_file.write_text(json.dumps(VALID_DATASET_JSON))
    newer_file.write_text(json.dumps(VALID_DATASET_JSON))

    result = discover_latest_golden_dataset(tmp_path, author=AUTHOR_A)

    assert result == newer_file


def test_discover_lexicographic_sort_limitation(tmp_path: Path) -> None:
    """Test documenting limitation: multi-digit minor versions don't sort semantically.

    Known limitation: Lexicographic sorting means v1.9 > v1.10 (semantically incorrect).
    This is acceptable for the use case and because the schema limits minor versions to
    single-digit decimals. If this becomes a problem, implement semantic version parsing.
    """
    # Create files demonstrating the limitation
    v1_9_filename = _golden_dataset_filename(AUTHOR_A, "1.9", DATE_OLD)
    v1_10_filename = _golden_dataset_filename(AUTHOR_A, "1.10", DATE_OLD)

    v1_9_file = tmp_path / v1_9_filename
    v1_10_file = tmp_path / v1_10_filename

    v1_9_file.write_text(json.dumps(VALID_DATASET_JSON))
    v1_10_file.write_text(json.dumps(VALID_DATASET_JSON))

    result = discover_latest_golden_dataset(tmp_path, author=AUTHOR_A)

    # Documents current behavior: v1.9 sorts after v1.10 lexicographically
    # Lexicographic: "v1.9" > "v1.10" (because '9' > '1' at position 4)
    assert result == v1_9_file  # v1.9 comes first (semantically wrong but expected)
    assert result != v1_10_file


def test_discover_no_matches(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset raises FileNotFoundError when no files match."""
    expected_pattern = f"{AUTHOR_A}_{DATASET_SCOPE}_v*.json"

    with pytest.raises(FileNotFoundError) as exc_info:
        discover_latest_golden_dataset(tmp_path, author=AUTHOR_A)

    error_message = str(exc_info.value)
    assert expected_pattern in error_message
    assert str(tmp_path) in error_message


def test_discover_respects_author(tmp_path: Path) -> None:
    """Test that discover_latest_golden_dataset filters by author parameter."""
    # Create dataset files for different authors
    author_a_filename = _golden_dataset_filename(AUTHOR_A, VERSION_OLD, DATE_OLD)
    author_b_filename = _golden_dataset_filename(AUTHOR_B, VERSION_OLD, DATE_OLD)

    author_a_file = tmp_path / author_a_filename
    author_b_file = tmp_path / author_b_filename

    author_a_file.write_text(json.dumps(VALID_DATASET_JSON))
    author_b_file.write_text(json.dumps(VALID_DATASET_JSON))

    result = discover_latest_golden_dataset(tmp_path, author=AUTHOR_A)

    assert result == author_a_file
    assert AUTHOR_A in result.name
    assert AUTHOR_B not in result.name


# --- Tests for save_eval_run() ---


def _create_minimal_eval_run(dataset_name: str = f"{AUTHOR_A}_golden") -> EvalRun:
    """Create a minimal EvalRun object for testing."""
    return EvalRun(
        dataset_version=VERSION_OLD,
        dataset_name=dataset_name,
        run_timestamp="2025-06-15T14:30:45+00:00",
        system_version={
            "chat_model": "test-model",
            "commit": "abc123def456"
        },
        example_results=[
            ExampleResult(
                example_id="test_example_001",
                question="Sample question?",
                language="en",
                response=ChatResponse(
                    text="Sample response",
                    retrieved_passage_ids=["chunk_abc123"],
                    retrieved_contexts=["Sample context text"],
                    retrieved_source_titles=["Sample Source"],
                    language="en"
                ),
                metrics=[
                    MetricResult(
                        name="test_metric",
                        score=0.85,
                        details={"status": "ok"}
                    )
                ],
                passed=True
            )
        ],
        aggregate_scores={
            "overall": {"test_metric": 0.85}
        },
        overall_pass_rate=1.0
    )


def test_save_creates_directory(tmp_path: Path) -> None:
    """Test that save_eval_run creates the output directory if it doesn't exist."""
    nonexistent_dir = tmp_path / "new_directory" / "nested"
    eval_run = _create_minimal_eval_run()

    # Directory should not exist yet
    assert not nonexistent_dir.exists()

    # Save should create the directory
    result_path = save_eval_run(eval_run, nonexistent_dir)

    # Directory should now exist
    assert nonexistent_dir.exists()
    assert nonexistent_dir.is_dir()
    assert result_path.parent == nonexistent_dir


def test_save_generates_valid_filename(tmp_path: Path) -> None:
    """Test that save_eval_run generates filename matching pattern {timestamp}.json."""
    eval_run = _create_minimal_eval_run(dataset_name=f"{AUTHOR_A}_golden")

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

    # Verify saved content can be loaded back as EvalRun
    saved_data = json.loads(result_path.read_text())
    loaded_eval_run = EvalRun(**saved_data)
    assert loaded_eval_run.dataset_name == eval_run.dataset_name
    assert loaded_eval_run.dataset_version == eval_run.dataset_version


def test_save_permission_error_on_directory_creation(tmp_path: Path) -> None:
    """Test that save_eval_run raises PermissionError with context when directory creation fails."""
    eval_run = _create_minimal_eval_run()

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


def test_save_permission_error_on_file_write(tmp_path: Path) -> None:
    """Test that save_eval_run raises PermissionError when file write is denied."""
    eval_run = _create_minimal_eval_run()

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
