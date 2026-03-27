"""Unit tests for vectorstore config module."""

from pathlib import Path

from src.configs.vectorstore_config import (
    COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_K,
    VECTOR_DB_PATH,
)


def test_collection_name_is_philosophes() -> None:
    """Test that COLLECTION_NAME is set to 'philosophes'."""
    assert COLLECTION_NAME == "philosophes"
    assert isinstance(COLLECTION_NAME, str)

def test_default_k_is_five() -> None:
    """Test that DEFAULT_K is set to 5."""
    assert DEFAULT_K == 5
    assert isinstance(DEFAULT_K, int)

def test_default_embedding_model_imported_correctly() -> None:
    """Test that DEFAULT_EMBEDDING_MODEL is re-exported from common config."""
    assert DEFAULT_EMBEDDING_MODEL == "nomic-embed-text"
    assert isinstance(DEFAULT_EMBEDDING_MODEL, str)

def test_vector_db_path_imported_correctly() -> None:
    """Test that VECTOR_DB_PATH is re-exported from common config."""
    assert VECTOR_DB_PATH == Path("data/chroma_db")
    assert isinstance(VECTOR_DB_PATH, Path)
