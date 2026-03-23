"""Unit tests for scripts/scrape_wikisource.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.configs.common import RAW_DATA_PATH
from src.configs.loader_configs import INGEST_CONFIGS

# Test constants
TEST_AUTHOR = "voltaire"
INVALID_AUTHOR = "nonexistent_author"
TEST_DOCUMENT_ID = "voltaire_lettres_philosophiques-1734"


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Test content page 1",
            metadata={
                "page_number": 1,
                "author": TEST_AUTHOR,
                "document_id": TEST_DOCUMENT_ID
            }
        ),
        Document(
            page_content="Test content page 2",
            metadata={
                "page_number": 2,
                "author": TEST_AUTHOR,
                "document_id": TEST_DOCUMENT_ID
            }
        ),
    ]


@pytest.fixture
def mock_loader(sample_documents: list[Document]) -> MagicMock:
    """Create a mock WikisourceLoader."""
    loader = MagicMock()
    loader.load.return_value = sample_documents
    return loader


@pytest.fixture
def mock_save_paths() -> list[Path]:
    """Create mock saved file paths."""
    return [
        Path(f"test/output/{TEST_DOCUMENT_ID}/page_01.json"),
        Path(f"test/output/{TEST_DOCUMENT_ID}/page_02.json"),
    ]


class TestScrapWikisourceMain:
    """Test the main() function of scrape_wikisource script."""

    @patch("scripts.scrape_wikisource.scrape_author")
    def test_default_arguments(
        self,
        mock_scrape: MagicMock,
    ) -> None:
        """Test main() with default arguments."""
        with patch("sys.argv", ["scrape_wikisource.py"]):
            from scripts.scrape_wikisource import main
            main()

        # Verify scrape_author called for all authors (since default author is None)
        assert mock_scrape.call_count == len(INGEST_CONFIGS)

        # Verify default output directory used
        call_args = mock_scrape.call_args[0]
        assert call_args[1] == str(RAW_DATA_PATH)

    @patch("scripts.scrape_wikisource.save_documents_to_disk")
    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.argv", ["scrape_wikisource.py", "--author", TEST_AUTHOR])
    def test_successful_scraping(
        self,
        mock_loader_class: MagicMock,
        mock_save: MagicMock,
        sample_documents: list[Document],
        mock_save_paths: list[Path]
    ) -> None:
        """Test successful scraping workflow."""
        from scripts.scrape_wikisource import main

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class.return_value = mock_loader_instance
        mock_save.return_value = mock_save_paths

        # Run main
        main()

        # Verify loader was instantiated with correct config
        mock_loader_class.assert_called_once()
        config = mock_loader_class.call_args[0][0]
        assert config.document_id == TEST_DOCUMENT_ID

        # Verify loader.load() was called
        mock_loader_instance.load.assert_called_once()

        # Verify save was called with correct arguments
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0]
        assert call_args[0] == sample_documents
        assert str(call_args[1]).endswith(TEST_DOCUMENT_ID)

    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.exit")
    @patch("sys.argv", ["scrape_wikisource.py", "--author", INVALID_AUTHOR])
    def test_invalid_author_exits_with_error(
        self,
        mock_exit: MagicMock,
        mock_loader_class: MagicMock
    ) -> None:
        """Test that invalid author causes script to exit with error."""
        from scripts.scrape_wikisource import main

        main()

        # Should exit with code 1
        mock_exit.assert_called_once_with(1)
        # Loader should not be instantiated
        mock_loader_class.assert_not_called()

    @patch("scripts.scrape_wikisource.save_documents_to_disk")
    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.argv", ["scrape_wikisource.py", "--author", TEST_AUTHOR])
    def test_no_documents_loaded_completes_successfully(
        self,
        mock_loader_class: MagicMock,
        mock_save: MagicMock
    ) -> None:
        """Test that no documents loaded completes without error (0 documents saved)."""
        from scripts.scrape_wikisource import main

        # Configure loader to return empty list
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_loader_class.return_value = mock_loader_instance

        main()

        # Should complete successfully without calling save
        mock_save.assert_not_called()

    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.exit")
    @patch("sys.argv", ["scrape_wikisource.py", "--author", TEST_AUTHOR])
    def test_loader_exception_exits_with_error(
        self,
        mock_exit: MagicMock,
        mock_loader_class: MagicMock
    ) -> None:
        """Test that exception during loading causes script to exit with error."""
        from scripts.scrape_wikisource import main

        # Configure loader to raise exception
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.side_effect = Exception("Network error")
        mock_loader_class.return_value = mock_loader_instance

        main()

        # Should exit with code 1
        mock_exit.assert_called_once_with(1)

    @patch("scripts.scrape_wikisource.save_documents_to_disk")
    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.exit")
    @patch("sys.argv", [
        "scrape_wikisource.py",
        "--author", TEST_AUTHOR,
        "--output-path", "custom/output"
    ])
    def test_custom_output_path(
        self,
        mock_exit: MagicMock,
        mock_loader_class: MagicMock,
        mock_save: MagicMock,
        sample_documents: list[Document]
    ) -> None:
        """Test that custom output directory is used."""
        from scripts.scrape_wikisource import main

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class.return_value = mock_loader_instance
        mock_save.return_value = [Path("custom/output/file.json")]

        main()

        # Verify save was called with custom directory
        mock_save.assert_called_once()
        output_path = mock_save.call_args[0][1]
        assert str(output_path).startswith("custom/output")
        assert str(output_path).endswith(TEST_DOCUMENT_ID)

    @patch("scripts.scrape_wikisource.save_documents_to_disk")
    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.exit")
    @patch("sys.argv", ["scrape_wikisource.py", "--author", TEST_AUTHOR])
    def test_save_exception_exits_with_error(
        self,
        mock_exit: MagicMock,
        mock_loader_class: MagicMock,
        mock_save: MagicMock,
        sample_documents: list[Document]
    ) -> None:
        """Test that exception during saving causes script to exit with error."""
        from scripts.scrape_wikisource import main

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class.return_value = mock_loader_instance
        mock_save.side_effect = Exception("Disk full")

        main()

        # Should exit with code 1
        mock_exit.assert_called_once_with(1)

    @patch("scripts.scrape_wikisource.save_documents_to_disk")
    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.argv", ["scrape_wikisource.py", "--author", TEST_AUTHOR])
    def test_output_path_construction(
        self,
        mock_loader_class: MagicMock,
        mock_save: MagicMock,
        sample_documents: list[Document]
    ) -> None:
        """Test that output directory path is correctly constructed."""
        from scripts.scrape_wikisource import main

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class.return_value = mock_loader_instance
        mock_save.return_value = [Path("file.json")]

        main()

        # Verify output path combines base dir with document_id
        output_path = mock_save.call_args[0][1]
        assert output_path == Path(RAW_DATA_PATH) / TEST_DOCUMENT_ID

    @patch("scripts.scrape_wikisource.save_documents_to_disk")
    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.argv", ["scrape_wikisource.py"])
    def test_default_scrapes_all_authors(
        self,
        mock_loader_class: MagicMock,
        mock_save: MagicMock,
        sample_documents: list[Document]
    ) -> None:
        """Test that running without --author scrapes all configured authors."""
        from scripts.scrape_wikisource import main

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class.return_value = mock_loader_instance
        mock_save.return_value = [Path("file.json")]

        main()

        # Should instantiate loader once per configured author
        assert mock_loader_class.call_count == len(INGEST_CONFIGS)
        # Should save once per author
        assert mock_save.call_count == len(INGEST_CONFIGS)

    @patch("scripts.scrape_wikisource.save_documents_to_disk")
    @patch("scripts.scrape_wikisource.WikisourceLoader")
    @patch("sys.argv", ["scrape_wikisource.py", "--output-path", "custom/dir"])
    def test_default_with_custom_output_path(
        self,
        mock_loader_class: MagicMock,
        mock_save: MagicMock,
        sample_documents: list[Document]
    ) -> None:
        """Test that default behavior works with custom output path."""
        from scripts.scrape_wikisource import main

        # Configure mocks
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class.return_value = mock_loader_instance
        mock_save.return_value = [Path("custom/dir/file.json")]

        main()

        # Verify save was called with custom directory
        output_path = mock_save.call_args[0][1]
        assert str(output_path).startswith("custom/dir")
