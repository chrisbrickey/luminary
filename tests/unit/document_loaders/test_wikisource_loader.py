"""Unit tests for WikisourceLoader."""

from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError, URLError

import pytest

from langchain_core.documents import Document

from src.configs.loader_configs import WikisourceCollectionConfig
from src.document_loaders.wikisource_loader import WikisourceLoader, _WikisourceHTMLExtractor
from tests.unit.document_loaders.conftest import create_api_response_bytes, create_mock_response


# Test data
SAMPLE_API_RESPONSE = {
    "parse": {
        "text": {
            "*": """
            <div class="mw-parser-output">
                <p>This is test content.</p>
                <div class="ws-noexport">Skip this</div>
                <p>More content here.</p>
                <script>console.log('skip');</script>
                <style>body { color: red; }</style>
            </div>
            """
        }
    }
}

ERROR_API_RESPONSE = {
    "error": {
        "code": "missingtitle",
        "info": "The page you requested doesn't exist."
    }
}


class TestWikisourceHTMLExtractor:
    """Test the HTML extraction parser."""

    def test_extracts_text_from_html(self) -> None:
        """Test that text is correctly extracted from HTML."""
        parser = _WikisourceHTMLExtractor()
        html = "<p>Hello</p><div>World</div>"
        parser.feed(html)
        assert parser.get_text() == "Hello World"

    def test_skips_script_and_style_tags(self) -> None:
        """Test that script and style tags are skipped."""
        parser = _WikisourceHTMLExtractor()
        html = """
            <p>Keep this</p>
            <script>console.log('skip');</script>
            <style>body { color: red; }</style>
            <p>Keep this too</p>
        """
        parser.feed(html)
        text = parser.get_text()
        assert "Keep this" in text
        assert "Keep this too" in text
        assert "console.log" not in text
        assert "color: red" not in text

    def test_skips_elements_with_skip_classes(self) -> None:
        """Test that elements with specific classes are skipped."""
        parser = _WikisourceHTMLExtractor()
        html = """
            <p>Keep this</p>
            <div class="ws-noexport">Skip this content</div>
            <div class="reference">Skip reference</div>
            <p>Keep this too</p>
        """
        parser.feed(html)
        text = parser.get_text()
        assert "Keep this" in text
        assert "Keep this too" in text
        assert "Skip this content" not in text
        assert "Skip reference" not in text

    def test_handles_nested_skip_elements(self) -> None:
        """Test that nested elements inside skip tags are also skipped."""
        parser = _WikisourceHTMLExtractor()
        html = """
            <p>Keep this</p>
            <div class="ws-noexport">
                <p>Skip this paragraph</p>
                <div>And this div</div>
            </div>
            <p>Keep this too</p>
        """
        parser.feed(html)
        text = parser.get_text()
        assert "Keep this" in text
        assert "Keep this too" in text
        assert "Skip this paragraph" not in text
        assert "And this div" not in text

    def test_handles_empty_html(self) -> None:
        """Test that empty HTML returns empty string."""
        parser = _WikisourceHTMLExtractor()
        parser.feed("")
        assert parser.get_text() == ""


class TestWikisourceLoader:
    """Test the WikisourceLoader class."""

    def test_load_successful(
        self,
        multi_page_config: WikisourceCollectionConfig,
        mock_sleep: Mock,
        mock_urlopen: Mock
    ) -> None:
        """Test successful loading of documents."""
        # Mock successful API responses
        mock_urlopen.return_value = create_mock_response(
            create_api_response_bytes(SAMPLE_API_RESPONSE)
        )

        loader = WikisourceLoader(multi_page_config)
        documents = loader.load()

        assert len(documents) == 2  # total_pages is 2
        assert all(doc.page_content for doc in documents)
        assert all('test_doc' in doc.metadata['document_id'] for doc in documents)
        assert documents[0].metadata['page_number'] == 1
        assert documents[1].metadata['page_number'] == 2
        # Verify delay between requests
        assert mock_sleep.call_count == 1  # Called between page 1 and 2

    def test_handles_missing_page(
        self,
        multi_page_config: WikisourceCollectionConfig,
        mock_sleep: Mock,
        mock_urlopen: Mock
    ) -> None:
        """Test handling of missing pages."""
        # First page exists, second doesn't
        responses = [
            create_api_response_bytes(SAMPLE_API_RESPONSE),
            create_api_response_bytes(ERROR_API_RESPONSE)
        ]

        mock_response = MagicMock()
        mock_response.read.side_effect = responses
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        loader = WikisourceLoader(multi_page_config)
        documents = loader.load()

        assert len(documents) == 1  # Only first page loaded
        assert documents[0].metadata['page_number'] == 1

    def test_retry_on_transient_error(
        self,
        minimal_config: WikisourceCollectionConfig,
        mock_sleep: Mock,
        mock_urlopen: Mock
    ) -> None:
        """Test retry logic for transient errors (429, 5xx)."""
        # First attempt fails with 429, second succeeds
        mock_urlopen.side_effect = [
            HTTPError("url", 429, "Too Many Requests", {}, None),
            create_mock_response(create_api_response_bytes(SAMPLE_API_RESPONSE))
        ]

        loader = WikisourceLoader(minimal_config, max_retries=2)
        documents = loader.load()

        assert len(documents) == 1
        # Should have called sleep for retry delay
        assert mock_sleep.call_count >= 1

    def test_retry_on_network_error(
        self,
        minimal_config: WikisourceCollectionConfig,
        mock_sleep: Mock,
        mock_urlopen: Mock
    ) -> None:
        """Test retry logic for network errors."""
        # First attempt fails with URLError, second succeeds
        mock_urlopen.side_effect = [
            URLError("Connection timeout"),
            create_mock_response(create_api_response_bytes(SAMPLE_API_RESPONSE))
        ]

        loader = WikisourceLoader(minimal_config, max_retries=2)
        documents = loader.load()

        assert len(documents) == 1

    def test_max_retries_exceeded(
        self,
        minimal_config: WikisourceCollectionConfig,
        mock_urlopen: Mock
    ) -> None:
        """Test that max retries is respected."""
        # All attempts fail with 500 error
        mock_urlopen.side_effect = HTTPError(
            "url", 500, "Internal Server Error", {}, None
        )

        loader = WikisourceLoader(minimal_config, max_retries=3, base_retry_delay=0.1)

        with pytest.raises(HTTPError):
            loader.load()

        # Should have tried max_retries times
        assert mock_urlopen.call_count == 3

    def test_non_transient_error_no_retry(
        self,
        minimal_config: WikisourceCollectionConfig,
        mock_urlopen: Mock
    ) -> None:
        """Test that non-transient errors (404) don't trigger retry."""
        # 404 error should not be retried
        mock_urlopen.side_effect = HTTPError(
            "url", 404, "Not Found", {}, None
        )

        loader = WikisourceLoader(minimal_config, max_retries=3)

        with pytest.raises(HTTPError) as exc_info:
            loader.load()

        assert exc_info.value.code == 404
        # Should have tried only once (no retry)
        assert mock_urlopen.call_count == 1

    def test_auto_discover_pages(
        self,
        auto_discover_config: WikisourceCollectionConfig,
        mock_sleep: Mock,
        mock_urlopen: Mock
    ) -> None:
        """Test auto-discovery of total pages."""
        # Pages 1-3 exist, page 4 doesn't
        responses = [
            # Discovery phase
            create_api_response_bytes(SAMPLE_API_RESPONSE),  # Page 1 exists
            create_api_response_bytes(SAMPLE_API_RESPONSE),  # Page 2 exists
            create_api_response_bytes(SAMPLE_API_RESPONSE),  # Page 3 exists
            create_api_response_bytes(ERROR_API_RESPONSE),   # Page 4 missing
            # Loading phase
            create_api_response_bytes(SAMPLE_API_RESPONSE),  # Load page 1
            create_api_response_bytes(SAMPLE_API_RESPONSE),  # Load page 2
            create_api_response_bytes(SAMPLE_API_RESPONSE),  # Load page 3
        ]

        mock_response = MagicMock()
        mock_response.read.side_effect = responses
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        loader = WikisourceLoader(auto_discover_config)
        documents = loader.load()

        assert len(documents) == 3

    def test_empty_response_handling(
        self,
        multi_page_config: WikisourceCollectionConfig,
        mock_urlopen: Mock
    ) -> None:
        """Test handling of empty API responses."""
        empty_response = {"parse": {"text": {"*": ""}}}
        mock_urlopen.return_value = create_mock_response(
            create_api_response_bytes(empty_response)
        )

        loader = WikisourceLoader(multi_page_config)
        documents = loader.load()

        assert len(documents) == 0  # No documents with content

    def test_metadata_construction(
        self,
        detailed_config: WikisourceCollectionConfig
    ) -> None:
        """Test that metadata is correctly constructed."""
        with patch('src.document_loaders.wikisource_loader.urlopen') as mock_urlopen:
            mock_urlopen.return_value = create_mock_response(
                create_api_response_bytes(SAMPLE_API_RESPONSE)
            )

            loader = WikisourceLoader(detailed_config)
            documents = loader.load()

            assert len(documents) == 1
            metadata = documents[0].metadata
            assert metadata['document_id'] == "test_id"
            assert metadata['document_title'] == "Test Title"
            assert metadata['author'] == "test_author"
            assert metadata['page_number'] == 1
            assert 'source' in metadata
            assert 'test.wikisource.org' in metadata['source']