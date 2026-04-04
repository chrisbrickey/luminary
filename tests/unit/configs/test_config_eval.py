"""Unit tests for eval config module."""

from pathlib import Path

from src.configs.eval import (
    DEFAULT_EVAL_ARTIFACTS_PATH,
    DEFAULT_GOLDEN_DATASET_PATH,
)


def test_golden_dataset_path_is_path_object() -> None:
    """Test that DEFAULT_GOLDEN_DATASET_PATH is a Path instance."""
    assert isinstance(DEFAULT_GOLDEN_DATASET_PATH, Path)


def test_golden_dataset_path_is_relative() -> None:
    """Test that DEFAULT_GOLDEN_DATASET_PATH is relative for portability."""
    assert not DEFAULT_GOLDEN_DATASET_PATH.is_absolute()


def test_golden_dataset_path_value() -> None:
    """Test that DEFAULT_GOLDEN_DATASET_PATH points to data/raw/golden."""
    assert DEFAULT_GOLDEN_DATASET_PATH == Path("data/raw/golden")


def test_eval_artifacts_path_is_path_object() -> None:
    """Test that DEFAULT_EVAL_ARTIFACTS_PATH is a Path instance."""
    assert isinstance(DEFAULT_EVAL_ARTIFACTS_PATH, Path)


def test_eval_artifacts_path_is_relative() -> None:
    """Test that DEFAULT_EVAL_ARTIFACTS_PATH is relative for portability."""
    assert not DEFAULT_EVAL_ARTIFACTS_PATH.is_absolute()


def test_eval_artifacts_path_value() -> None:
    """Test that DEFAULT_EVAL_ARTIFACTS_PATH points to evals/runs."""
    assert DEFAULT_EVAL_ARTIFACTS_PATH == Path("evals/runs")
