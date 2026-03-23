"""Unit tests for vectorstore config module."""

from pathlib import Path

from src.configs.vectorstore_config import (
    COLLECTION_NAME,
    VECTOR_DB_PATH,
    DEFAULT_K,
    EMBEDDING_MODEL,
)


def test_collection_name_is_philosophes() -> None:
    """Test that COLLECTION_NAME is set to 'philosophes'."""
    assert COLLECTION_NAME == "philosophes"
    assert isinstance(COLLECTION_NAME, str)


def test_embedding_model_is_nomic_embed_text() -> None:
    """Test that EMBEDDING_MODEL is set to 'nomic-embed-text'."""
    assert EMBEDDING_MODEL == "nomic-embed-text"
    assert isinstance(EMBEDDING_MODEL, str)


def test_default_k_is_five() -> None:
    """Test that DEFAULT_K is set to 5."""
    assert DEFAULT_K == 5
    assert isinstance(DEFAULT_K, int)


def test_vector_db_path_imported_correctly() -> None:
    """Test that VECTOR_DB_PATH is re-exported from common config."""
    assert VECTOR_DB_PATH == Path("data/chroma_db")
    assert isinstance(VECTOR_DB_PATH, Path)
