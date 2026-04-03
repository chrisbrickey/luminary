"""Unit tests for eval utility functions."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.utils import discover_latest_golden_dataset, load_golden_dataset
from src.schemas.eval import GoldenDataset

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
