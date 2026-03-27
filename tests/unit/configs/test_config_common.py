"""Unit tests for common config module."""

from pathlib import Path

from src.configs.common import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    VECTOR_DB_PATH,
)


def test_vector_db_path_is_path_instance() -> None:
    """Test that VECTOR_DB_PATH is a Path instance."""
    assert isinstance(VECTOR_DB_PATH, Path)


def test_vector_db_path_points_to_chroma_db() -> None:
    """Test that VECTOR_DB_PATH points to data/chroma_db."""
    assert VECTOR_DB_PATH == Path("data/chroma_db")
    assert VECTOR_DB_PATH.name == "chroma_db"
    assert VECTOR_DB_PATH.parent == Path("data")


def test_default_chat_model_is_string() -> None:
    """Test that DEFAULT_CHAT_MODEL is a string."""
    assert isinstance(DEFAULT_CHAT_MODEL, str)


def test_default_chat_model_is_mistral() -> None:
    """Test that DEFAULT_CHAT_MODEL is set to 'mistral'."""
    assert DEFAULT_CHAT_MODEL == "mistral"


def test_default_embedding_model_is_string() -> None:
    """Test that DEFAULT_EMBEDDING_MODEL is a string."""
    assert isinstance(DEFAULT_EMBEDDING_MODEL, str)


def test_default_embedding_model_is_nomic_embed_text() -> None:
    """Test that DEFAULT_EMBEDDING_MODEL is set to 'nomic-embed-text'."""
    assert DEFAULT_EMBEDDING_MODEL == "nomic-embed-text"
