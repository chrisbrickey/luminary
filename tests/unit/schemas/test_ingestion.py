"""Tests for src/schemas/ingestion.py - data ingestion and document loading schemas"""

from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.ingestion import WikisourceCollection

# --- Shared test constants ---

DOCUMENT_ID = "test-document-id"
DOCUMENT_TITLE = "Test Document Title"
AUTHOR = "test_author"
PAGE_TITLE_TEMPLATE = "Test Document/Page {n}"
TOTAL_PAGES = 42
CUSTOM_API_URL = "https://en.wikisource.org/w/api.php"
DEFAULT_API_URL = "https://fr.wikisource.org/w/api.php"


def _collection_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return default WikisourceCollection kwargs, with optional overrides."""
    defaults: dict[str, Any] = {
        "document_id": DOCUMENT_ID,
        "document_title": DOCUMENT_TITLE,
        "author": AUTHOR,
        "page_title_template": PAGE_TITLE_TEMPLATE,
    }
    defaults.update(overrides)
    return defaults


class TestWikisourceCollection:
    def test_construction_with_required_fields(self) -> None:
        collection = WikisourceCollection(**_collection_kwargs())
        assert collection.document_id == DOCUMENT_ID
        assert collection.document_title == DOCUMENT_TITLE
        assert collection.author == AUTHOR
        assert collection.page_title_template == PAGE_TITLE_TEMPLATE
        assert collection.total_pages is None
        assert collection.api_url == DEFAULT_API_URL

    def test_construction_with_all_fields(self) -> None:
        collection = WikisourceCollection(
            **_collection_kwargs(
                total_pages=TOTAL_PAGES,
                api_url=CUSTOM_API_URL,
            )
        )
        assert collection.document_id == DOCUMENT_ID
        assert collection.document_title == DOCUMENT_TITLE
        assert collection.author == AUTHOR
        assert collection.page_title_template == PAGE_TITLE_TEMPLATE
        assert collection.total_pages == TOTAL_PAGES
        assert collection.api_url == CUSTOM_API_URL

    def test_total_pages_defaults_to_none(self) -> None:
        collection = WikisourceCollection(**_collection_kwargs())
        assert collection.total_pages is None

    def test_api_url_defaults_to_french_wikisource(self) -> None:
        collection = WikisourceCollection(**_collection_kwargs())
        assert collection.api_url == DEFAULT_API_URL

    def test_custom_api_url(self) -> None:
        collection = WikisourceCollection(
            **_collection_kwargs(api_url=CUSTOM_API_URL)
        )
        assert collection.api_url == CUSTOM_API_URL

    def test_total_pages_explicit_none(self) -> None:
        collection = WikisourceCollection(**_collection_kwargs(total_pages=None))
        assert collection.total_pages is None

    def test_total_pages_zero(self) -> None:
        collection = WikisourceCollection(**_collection_kwargs(total_pages=0))
        assert collection.total_pages == 0

    def test_total_pages_positive(self) -> None:
        collection = WikisourceCollection(**_collection_kwargs(total_pages=100))
        assert collection.total_pages == 100

    def test_missing_document_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _collection_kwargs()
            del kwargs["document_id"]
            WikisourceCollection(**kwargs)  # type: ignore[arg-type]

    def test_missing_document_title_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _collection_kwargs()
            del kwargs["document_title"]
            WikisourceCollection(**kwargs)  # type: ignore[arg-type]

    def test_missing_author_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _collection_kwargs()
            del kwargs["author"]
            WikisourceCollection(**kwargs)  # type: ignore[arg-type]

    def test_missing_page_title_template_raises(self) -> None:
        with pytest.raises(ValidationError):
            kwargs = _collection_kwargs()
            del kwargs["page_title_template"]
            WikisourceCollection(**kwargs)  # type: ignore[arg-type]

    def test_document_id_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            WikisourceCollection(**_collection_kwargs(document_id=123))  # type: ignore[arg-type]

    def test_document_title_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            WikisourceCollection(**_collection_kwargs(document_title=123))  # type: ignore[arg-type]

    def test_author_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            WikisourceCollection(**_collection_kwargs(author=123))  # type: ignore[arg-type]

    def test_page_title_template_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            WikisourceCollection(**_collection_kwargs(page_title_template=123))  # type: ignore[arg-type]

    def test_total_pages_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            WikisourceCollection(**_collection_kwargs(total_pages="not-an-int"))  # type: ignore[arg-type]

    def test_api_url_type_enforced(self) -> None:
        with pytest.raises(ValidationError):
            WikisourceCollection(**_collection_kwargs(api_url=123))  # type: ignore[arg-type]

    def test_empty_string_document_id_allowed(self) -> None:
        # Pydantic allows empty strings for str fields by default
        collection = WikisourceCollection(**_collection_kwargs(document_id=""))
        assert collection.document_id == ""

    def test_page_title_template_with_placeholder(self) -> None:
        template = "Book/Chapter {n}"
        collection = WikisourceCollection(
            **_collection_kwargs(page_title_template=template)
        )
        assert collection.page_title_template == template
        assert "{n}" in collection.page_title_template
