"""Unit tests for eval config module."""

from pathlib import Path

from src.configs.eval import DEFAULT_GOLDEN_DATASET_PATH


def test_golden_dataset_path_is_path_object() -> None:
    """Test that DEFAULT_GOLDEN_DATASET_PATH is a Path instance."""
    assert isinstance(DEFAULT_GOLDEN_DATASET_PATH, Path)


def test_golden_dataset_path_points_to_golden_dir() -> None:
    """Test that DEFAULT_GOLDEN_DATASET_PATH points to evals/golden."""
    assert DEFAULT_GOLDEN_DATASET_PATH == Path("evals/golden")
    assert DEFAULT_GOLDEN_DATASET_PATH.name == "golden"
    assert DEFAULT_GOLDEN_DATASET_PATH.parent == Path("evals")
