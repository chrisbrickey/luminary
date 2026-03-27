"""Integration tests that verify WikisourceLoader works with the external Wikisource API.

IMPORTANT: These tests make real HTTP calls to Wikisource.
- Requires internet connection
- May be slow (~1-2 seconds per test)
- Run with: uv run pytest -m external
- Excluded from default test run
"""

import pytest
from langchain_core.documents import Document

from src.configs.loader_configs import LETTRES_PHILOSOPHIQUES
from src.document_loaders.wikisource_loader import WikisourceLoader


@pytest.fixture(scope="module")
def sample_document() -> Document:
    """Fetch only the first page to keep the test fast."""
    collection = LETTRES_PHILOSOPHIQUES.model_copy(update={"total_pages": 1})
    documents = WikisourceLoader(collection, delay=0).load()
    return documents[0]


@pytest.mark.integration
@pytest.mark.external
def test_load_returns_non_empty_content(sample_document: Document) -> None:
    """Verify that real API calls return substantial content."""
    assert len(sample_document.page_content) > 200


@pytest.mark.integration
@pytest.mark.external
def test_load_document_contains_expected_language_content(sample_document: Document) -> None:
    """Verify content is in expected language and contains relevant terms.

    Lettre 1 is 'Sur les Quakers' - at least one of these terms must appear
    in any correct parse of the first letter.
    """
    content = sample_document.page_content.upper()
    # At least one of these should appear in the first letter
    assert any(term in content for term in ["QUAKERS", "LIBERTÉ", "RELIGION", "ANGLAIS"])


@pytest.mark.integration
@pytest.mark.external
def test_load_document_has_no_html_tags(sample_document: Document) -> None:
    """Verify HTML parser successfully strips all tags."""
    content = sample_document.page_content
    assert "<" not in content
    assert ">" not in content
    # Also check no script/style content leaked through
    assert "console.log" not in content.lower()
    # CSS leak check - allow natural braces in text but not CSS patterns
    assert "color:" not in content.lower()


@pytest.mark.integration
@pytest.mark.external
def test_load_document_has_no_excluded_wikisource_markup(sample_document: Document) -> None:
    """Verify Wikisource-specific markup classes are filtered out."""
    content = sample_document.page_content.lower()
    # These shouldn't appear if ws-noexport/reference filtering works
    assert "ws-noexport" not in content


@pytest.mark.integration
@pytest.mark.external
def test_load_document_metadata_is_correct(sample_document: Document) -> None:
    """Verify metadata fields are populated correctly from real API."""
    meta = sample_document.metadata
    assert meta["document_id"] == "voltaire_lettres_philosophiques-1734"
    assert meta["document_title"] == "Lettres Philosophiques 1734"
    assert meta["author"] == "voltaire"
    assert meta["page_number"] == 1
    assert "source" in meta
    assert "fr.wikisource.org" in meta["source"]


@pytest.mark.integration
@pytest.mark.external
def test_load_multiple_pages() -> None:
    """Verify loading multiple pages works and respects delays."""
    collection = LETTRES_PHILOSOPHIQUES.model_copy(update={"total_pages": 2})
    documents = WikisourceLoader(collection, delay=0.1).load()

    assert len(documents) == 2
    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2
    # Content should be different
    assert documents[0].page_content != documents[1].page_content


@pytest.mark.integration
@pytest.mark.external
def test_handles_nonexistent_page() -> None:
    """Verify graceful handling when a page doesn't exist."""
    collection = LETTRES_PHILOSOPHIQUES.model_copy(
        update={
            "page_title_template": "NonExistent/Page_{n}",
            "total_pages": 1
        }
    )
    documents = WikisourceLoader(collection, delay=0).load()

    # Should return empty list, not crash
    assert len(documents) == 0
