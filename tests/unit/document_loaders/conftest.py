"""Shared fixtures and helpers for document loader tests."""

import json
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.schemas import WikisourceCollection


# Helper functions for creating mock responses
def create_mock_response(content: bytes) -> MagicMock:
    """Create a mock HTTP response with the given content."""
    mock_response = MagicMock()
    mock_response.read.return_value = content
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__ = lambda *args: None
    return mock_response


def create_api_response_bytes(data: dict) -> bytes:
    """Convert API response dict to encoded bytes."""
    return json.dumps(data).encode()


# Fixtures for mock patches
@pytest.fixture
def mock_urlopen() -> Generator[Mock, None, None]:
    """Mock urlopen for all WikisourceLoader tests."""
    with patch('src.document_loaders.wikisource_loader.urlopen') as mock:
        yield mock


@pytest.fixture
def mock_sleep() -> Generator[Mock, None, None]:
    """Mock time.sleep to speed up tests."""
    with patch('src.document_loaders.wikisource_loader.time.sleep') as mock:
        yield mock


# Fixtures for common config variations
@pytest.fixture
def minimal_config() -> WikisourceCollection:
    """Minimal config with 1 page for testing."""
    return WikisourceCollection(
        document_id="test_doc",
        document_title="Test",
        author="test_author",
        page_title_template="Test/{n}",
        total_pages=1
    )


@pytest.fixture
def multi_page_config() -> WikisourceCollection:
    """Config with 2 pages for testing."""
    return WikisourceCollection(
        document_id="test_doc",
        document_title="Test Document",
        author="test_author",
        page_title_template="Test/Page {n}",
        total_pages=2,
        api_url="https://test.wikisource.org/w/api.php"
    )


@pytest.fixture
def auto_discover_config() -> WikisourceCollection:
    """Config with auto-discovery enabled (total_pages=None)."""
    return WikisourceCollection(
        document_id="test_doc",
        document_title="Test",
        author="test_author",
        page_title_template="Test/{n}",
        total_pages=None
    )


@pytest.fixture
def detailed_config() -> WikisourceCollection:
    """Config with all fields specified for metadata testing."""
    return WikisourceCollection(
        document_id="test_id",
        document_title="Test Title",
        author="test_author",
        page_title_template="Test/{n}",
        total_pages=1,
        api_url="https://test.wikisource.org/w/api.php"
    )
