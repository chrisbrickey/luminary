"""Unit tests for common config module."""

from pathlib import Path

from src.configs.common import DEFAULT_DB_PATH


def test_default_db_path_is_path_instance() -> None:
    """Test that DEFAULT_DB_PATH is a Path instance."""
    assert isinstance(DEFAULT_DB_PATH, Path)


def test_default_db_path_points_to_chroma_db() -> None:
    """Test that DEFAULT_DB_PATH points to data/chroma_db."""
    assert DEFAULT_DB_PATH == Path("data/chroma_db")
    assert DEFAULT_DB_PATH.name == "chroma_db"
    assert DEFAULT_DB_PATH.parent == Path("data")
