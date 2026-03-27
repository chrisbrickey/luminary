"""Unit tests for loader_configs module."""

from pathlib import Path

from src.configs.common import VECTOR_DB_PATH
from src.configs.loader_configs import (
    INGEST_CONFIGS,
    LETTRES_PHILOSOPHIQUES,
)
from src.schemas import WikisourceCollection


def test_vector_db_path_is_correct() -> None:
    """Test that VECTOR_DB_PATH points to the correct location."""
    assert VECTOR_DB_PATH == Path("data/chroma_db")
    assert VECTOR_DB_PATH.name == "chroma_db"
    assert VECTOR_DB_PATH.parent.name == "data"

def test_ingest_configs_has_voltaire_key() -> None:
    """Test that INGEST_CONFIGS registry has voltaire key."""
    assert "voltaire" in INGEST_CONFIGS
    assert INGEST_CONFIGS["voltaire"] == LETTRES_PHILOSOPHIQUES


def test_wikisource_collection_default_api_url() -> None:
    """Test that WikisourceCollection has default API URL."""
    collection = WikisourceCollection(
        document_id="test_id",
        document_title="Test Title",
        author="test_author",
        page_title_template="Test/Page {n}",
    )
    assert collection.api_url == "https://fr.wikisource.org/w/api.php"


def test_wikisource_collection_optional_total_pages() -> None:
    """Test that total_pages is optional and defaults to None."""
    collection = WikisourceCollection(
        document_id="test_id",
        document_title="Test Title",
        author="test_author",
        page_title_template="Test/Page {n}",
    )
    assert collection.total_pages is None


def test_wikisource_collection_custom_api_url() -> None:
    """Test that custom API URL can be set."""
    custom_url = "https://en.wikisource.org/w/api.php"
    collection = WikisourceCollection(
        document_id="test_id",
        document_title="Test Title",
        author="test_author",
        page_title_template="Test/Page {n}",
        api_url=custom_url,
    )
    assert collection.api_url == custom_url

def test_lettres_philosophiques_has_required_fields() -> None:
    """Test that LETTRES_PHILOSOPHIQUES has all required fields."""
    assert LETTRES_PHILOSOPHIQUES.document_id == "voltaire_lettres_philosophiques-1734"
    assert LETTRES_PHILOSOPHIQUES.document_title == "Lettres Philosophiques 1734"
    assert LETTRES_PHILOSOPHIQUES.author == "voltaire" # lowercase
    assert LETTRES_PHILOSOPHIQUES.page_title_template == "Lettres philosophiques/Lettre {n}"
    assert LETTRES_PHILOSOPHIQUES.total_pages is None  # auto-discovery
    assert LETTRES_PHILOSOPHIQUES.api_url == "https://fr.wikisource.org/w/api.php"