"""Unit tests for loader_configs module."""

from pathlib import Path

from src.configs.common import DEFAULT_DB_PATH
from src.configs.loader_configs import (
    INGEST_CONFIGS,
    LETTRES_PHILOSOPHIQUES_CONFIG,
    WikisourceCollectionConfig,
)


def test_default_db_path_is_correct() -> None:
    """Test that DEFAULT_DB_PATH points to the correct location."""
    assert DEFAULT_DB_PATH == Path("data/chroma_db")
    assert DEFAULT_DB_PATH.name == "chroma_db"
    assert DEFAULT_DB_PATH.parent.name == "data"

def test_ingest_configs_has_voltaire_key() -> None:
    """Test that INGEST_CONFIGS registry has voltaire key."""
    assert "voltaire" in INGEST_CONFIGS
    assert INGEST_CONFIGS["voltaire"] == LETTRES_PHILOSOPHIQUES_CONFIG


def test_wikisource_config_default_api_url() -> None:
    """Test that WikisourceCollectionConfig has default API URL."""
    config = WikisourceCollectionConfig(
        document_id="test_id",
        document_title="Test Title",
        author="test_author",
        page_title_template="Test/Page {n}",
    )
    assert config.api_url == "https://fr.wikisource.org/w/api.php"


def test_wikisource_config_optional_total_pages() -> None:
    """Test that total_pages is optional and defaults to None."""
    config = WikisourceCollectionConfig(
        document_id="test_id",
        document_title="Test Title",
        author="test_author",
        page_title_template="Test/Page {n}",
    )
    assert config.total_pages is None


def test_wikisource_config_custom_api_url() -> None:
    """Test that custom API URL can be set."""
    custom_url = "https://en.wikisource.org/w/api.php"
    config = WikisourceCollectionConfig(
        document_id="test_id",
        document_title="Test Title",
        author="test_author",
        page_title_template="Test/Page {n}",
        api_url=custom_url,
    )
    assert config.api_url == custom_url

def test_lettres_philosophiques_config_has_required_fields() -> None:
    """Test that LETTRES_PHILOSOPHIQUES_CONFIG has all required fields."""
    assert LETTRES_PHILOSOPHIQUES_CONFIG.document_id == "voltaire_lettres_philosophiques-1734"
    assert LETTRES_PHILOSOPHIQUES_CONFIG.document_title == "Lettres Philosophiques 1734"
    assert LETTRES_PHILOSOPHIQUES_CONFIG.author == "voltaire" # lowercase
    assert LETTRES_PHILOSOPHIQUES_CONFIG.page_title_template == "Lettres philosophiques/Lettre {n}"
    assert LETTRES_PHILOSOPHIQUES_CONFIG.total_pages is None  # auto-discovery
    assert LETTRES_PHILOSOPHIQUES_CONFIG.api_url == "https://fr.wikisource.org/w/api.php"