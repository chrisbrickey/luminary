"""Loader for fetching texts from Wikisource."""

import json
import logging
import time
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from langchain_core.documents import Document

from src.configs.loader_configs import WikisourceCollectionConfig

logger = logging.getLogger(__name__)


class _WikisourceHTMLExtractor(HTMLParser):
    """HTML parser that extracts text content while skipping certain elements."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.skip_depth = 0
        self.current_tag: str | None = None
        # Tags to skip entirely (including all their children)
        self.skip_tags = {
            'script', 'style', 'noscript'
        }
        # Classes to skip
        self.skip_classes = {
            'ws-noexport', 'reference', 'reflist', 'references',
            'mw-editsection', 'noprint', 'navbox'
        }

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.current_tag = tag

        # Check if we should skip this element
        if self.skip_depth > 0:
            self.skip_depth += 1
            return

        # Skip specific tags
        if tag in self.skip_tags:
            self.skip_depth = 1
            return

        # Skip elements with certain classes
        for attr_name, attr_value in attrs:
            if attr_name == 'class' and attr_value:
                classes = set(attr_value.split())
                if classes & self.skip_classes:
                    self.skip_depth = 1
                    return

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth > 0:
            self.skip_depth -= 1
        self.current_tag = None

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            # Skip whitespace-only text between block elements
            if data.strip():
                self.text_parts.append(data)

    def get_text(self) -> str:
        """Get the extracted text content."""
        # Join parts and clean up excessive whitespace
        text = ' '.join(self.text_parts)
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        return text.strip()


class WikisourceLoader:
    """Loads documents from Wikisource using the MediaWiki API."""

    def __init__(
        self,
        config: WikisourceCollectionConfig,
        delay: float = 1.0,
        max_retries: int = 3,
        base_retry_delay: float = 2.0
    ):
        """Initialize the loader.

        Args:
            config: Configuration for the collection to load
            delay: Delay in seconds between API requests (be polite!)
            max_retries: Maximum number of retry attempts for transient errors
            base_retry_delay: Base delay for exponential backoff on retries
        """
        self.config = config
        self.delay = delay
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

    def _fetch_page_html(self, page_title: str) -> str:
        """Fetch the HTML content of a single Wikisource page.

        Args:
            page_title: The title of the page to fetch

        Returns:
            The HTML content of the page

        Raises:
            HTTPError: For non-transient HTTP errors
            URLError: For network-related errors after all retries
        """
        params = {
            'action': 'parse',
            'format': 'json',
            'page': page_title,
            'prop': 'text',
            'disableeditsection': 'true',
            'disabletoc': 'true'
        }

        url = f"{self.config.api_url}?{urlencode(params)}"

        # Add User-Agent header per Wikimedia API etiquette
        headers = {
            'User-Agent': 'Luminary/0.1.0 (Educational RAG Project; Python/urllib)'
        }

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Fetching page: {page_title} (attempt {attempt + 1})")

                request = Request(url, headers=headers)
                with urlopen(request, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))

                if 'error' in data:
                    # API returned an error (e.g., page not found)
                    error_code = data['error'].get('code', 'unknown')
                    error_info = data['error'].get('info', 'Unknown error')

                    if error_code == 'missingtitle':
                        # Page doesn't exist - not a transient error
                        logger.warning(f"Page not found: {page_title}")
                        return ""
                    else:
                        raise ValueError(f"API error for {page_title}: {error_info}")

                # Extract HTML from response
                html: str = str(data.get('parse', {}).get('text', {}).get('*', ''))
                return html

            except (HTTPError, URLError, TimeoutError, ConnectionError) as e:
                # Check if it's a transient error
                is_transient = False

                if isinstance(e, HTTPError):
                    # 429 Too Many Requests or 5xx errors are transient
                    if e.code == 429 or e.code >= 500:
                        is_transient = True
                elif isinstance(e, (URLError, TimeoutError, ConnectionError)):
                    # Network errors are transient
                    is_transient = True

                if is_transient and attempt < self.max_retries - 1:
                    # Exponential backoff
                    retry_delay = self.base_retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Transient error fetching {page_title}: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    # Non-transient error or max retries reached
                    logger.error(f"Failed to fetch {page_title}: {e}")
                    raise

        # Should not reach here
        raise RuntimeError(f"Failed to fetch {page_title} after {self.max_retries} attempts")

    def _parse_html_to_text(self, html: str) -> str:
        """Parse HTML content to extract plain text.

        Args:
            html: The HTML content to parse

        Returns:
            The extracted plain text
        """
        if not html:
            return ""

        parser = _WikisourceHTMLExtractor()
        parser.feed(html)
        return parser.get_text()

    def _discover_total_pages(self) -> int:
        """Auto-discover the total number of pages if not specified.

        Returns:
            The total number of pages found
        """
        # Start with a reasonable guess and probe
        page_num = 1
        last_valid = 0

        # Binary search would be more efficient, but simple linear probe is fine
        # for collections with < 100 pages
        while page_num <= 100:  # Reasonable upper limit
            page_title = self.config.page_title_template.format(n=page_num)
            try:
                html = self._fetch_page_html(page_title)
                if html:  # Page exists
                    last_valid = page_num
                    page_num += 1
                else:
                    # Page doesn't exist, we've found the limit
                    break
            except Exception:
                # Assume we've reached the end
                break

            # Be polite between requests
            time.sleep(self.delay)

        logger.info(f"Auto-discovered {last_valid} pages for {self.config.document_id}")
        return last_valid

    def load(self) -> list[Document]:
        """Load all pages from the configured Wikisource collection.

        Returns:
            List of Document objects with content and metadata
        """
        # Determine total pages
        if self.config.total_pages is not None:
            total_pages = self.config.total_pages
        else:
            total_pages = self._discover_total_pages()

        if total_pages == 0:
            logger.warning(f"No pages found for {self.config.document_id}")
            return []

        documents = []

        for page_num in range(1, total_pages + 1):
            # Construct page title
            page_title = self.config.page_title_template.format(n=page_num)

            # Fetch HTML
            html = self._fetch_page_html(page_title)

            # Parse to text
            text = self._parse_html_to_text(html)

            if text:  # Only add non-empty documents
                # Construct source URL
                base_url = self.config.api_url.replace('/w/api.php', '')
                source_url = urljoin(base_url, f'/wiki/{page_title.replace(" ", "_")}')

                # Create Document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        'document_id': self.config.document_id,
                        'document_title': self.config.document_title,
                        'author': self.config.author,
                        'source': source_url,
                        'page_number': page_num
                    }
                )
                documents.append(doc)

                logger.info(
                    f"Loaded page {page_num}/{total_pages} "
                    f"for {self.config.document_id} "
                    f"({len(text)} chars)"
                )
            else:
                logger.warning(
                    f"Page {page_num} for {self.config.document_id} "
                    f"was empty or not found"
                )

            # Be polite between requests (except for the last one)
            if page_num < total_pages:
                time.sleep(self.delay)

        logger.info(
            f"Successfully loaded {len(documents)} documents "
            f"for {self.config.document_id}"
        )

        return documents