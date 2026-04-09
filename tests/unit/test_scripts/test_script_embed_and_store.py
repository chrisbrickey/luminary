"""Unit tests for scripts/embed_and_store.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.configs.common import RAW_DATA_PATH
from src.configs.loader_configs import INGEST_CONFIGS
from src.configs.vectorstore_config import COLLECTION_NAME

# Test constants
TEST_AUTHOR = "voltaire"
INVALID_AUTHOR = "nonexistent_author"
TEST_DOCUMENT_ID = "voltaire_lettres_philosophiques-1734"

# Test data for helper function tests (not testing defaults)
TEST_INPUT_PATH = "test/input"
TEST_DB_PATH = "test/db"


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
def sample_chunks() -> list[Document]:
    """Create sample chunks for testing."""
    return [
        Document(
            page_content="Test chunk 1",
            metadata={
                "chunk_id": "abc123",
                "chunk_index": 0,
                "document_id": TEST_DOCUMENT_ID,
                "author": TEST_AUTHOR
            }
        ),
        Document(
            page_content="Test chunk 2",
            metadata={
                "chunk_id": "def456",
                "chunk_index": 1,
                "document_id": TEST_DOCUMENT_ID,
                "author": TEST_AUTHOR
            }
        ),
    ]


class TestEmbedAndStoreMain:
    """Test main() function of embed_and_store script."""

    @patch("scripts.embed_and_store.check_ollama_or_exit")
    @patch("scripts.embed_and_store.embed_author")
    def test_default_arguments(
        self,
        mock_embed: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test main() with default arguments."""
        mock_ollama.return_value = None
        mock_embed.return_value = 5  # Mock chunk count

        with patch("sys.argv", ["embed_and_store.py"]):
            from scripts.embed_and_store import main
            main()

        # Verify Ollama check was called
        mock_ollama.assert_called_once()

        # Verify embed_author called for all authors (since default author is None)
        assert mock_embed.call_count == len(INGEST_CONFIGS)

        # Verify default directories used
        call_args = mock_embed.call_args[0]
        assert call_args[1] == str(RAW_DATA_PATH)  # input_base_path

    @patch("scripts.embed_and_store.check_ollama_or_exit")
    @patch("scripts.embed_and_store.embed_and_store")
    @patch("scripts.embed_and_store.chunk_documents")
    @patch("scripts.embed_and_store.load_documents_from_disk")
    def test_successful_embedding_single_author(
        self,
        mock_load: MagicMock,
        mock_chunk: MagicMock,
        mock_embed: MagicMock,
        mock_ollama: MagicMock,
        sample_documents: list[Document],
        sample_chunks: list[Document],
    ) -> None:
        """Test successful embedding for a single author."""
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        mock_ollama.return_value = None

        with patch("sys.argv", ["embed_and_store.py", "--author", TEST_AUTHOR]):
            from scripts.embed_and_store import main
            main()

        # Verify Ollama check was called
        mock_ollama.assert_called_once()

        # Verify load was called with correct default path
        expected_path = Path(RAW_DATA_PATH) / TEST_DOCUMENT_ID
        mock_load.assert_called_once_with(expected_path)

        # Verify chunk was called with documents
        mock_chunk.assert_called_once_with(sample_documents)

        # Verify embed_and_store was called (db path from environment)
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args[1]
        assert call_kwargs["chunks"] == sample_chunks
        assert call_kwargs["collection_name"] == COLLECTION_NAME

    @patch("scripts.embed_and_store.check_ollama_or_exit")
    @patch("scripts.embed_and_store.embed_and_store")
    @patch("scripts.embed_and_store.chunk_documents")
    @patch("scripts.embed_and_store.load_documents_from_disk")
    def test_default_processes_all_authors(
        self,
        mock_load: MagicMock,
        mock_chunk: MagicMock,
        mock_embed: MagicMock,
        mock_ollama: MagicMock,
        sample_documents: list[Document],
        sample_chunks: list[Document],
    ) -> None:
        """Test that no --author flag processes all configured authors."""
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        mock_ollama.return_value = None

        with patch("sys.argv", ["embed_and_store.py"]):
            from scripts.embed_and_store import main
            main()

        # Verify Ollama check was called
        mock_ollama.assert_called_once()

        # Should process all authors (currently just voltaire)
        assert mock_load.call_count == len(INGEST_CONFIGS)
        assert mock_chunk.call_count == len(INGEST_CONFIGS)
        assert mock_embed.call_count == len(INGEST_CONFIGS)

    @patch("scripts.embed_and_store.check_ollama_or_exit")
    @patch("scripts.embed_and_store.embed_and_store")
    @patch("scripts.embed_and_store.chunk_documents")
    @patch("scripts.embed_and_store.load_documents_from_disk")
    def test_custom_input_path(
        self,
        mock_load: MagicMock,
        mock_chunk: MagicMock,
        mock_embed: MagicMock,
        mock_ollama: MagicMock,
        sample_documents: list[Document],
        sample_chunks: list[Document],
    ) -> None:
        """Test using custom input directory."""
        custom_input = "custom/input"
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        mock_ollama.return_value = None

        with patch("sys.argv", [
            "embed_and_store.py",
            "--author", TEST_AUTHOR,
            "--input-path", custom_input
        ]):
            from scripts.embed_and_store import main
            main()

        # Verify load was called with custom input path
        expected_path = Path(custom_input) / TEST_DOCUMENT_ID
        mock_load.assert_called_once_with(expected_path)

    @patch("scripts.embed_and_store.check_ollama_or_exit")
    def test_invalid_author_exits_with_error(
        self,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that invalid author exits with error."""
        mock_ollama.return_value = None

        with patch("sys.argv", ["embed_and_store.py", "--author", INVALID_AUTHOR]):
            from scripts.embed_and_store import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("scripts.embed_and_store.check_ollama_or_exit")
    @patch("scripts.embed_and_store.load_documents_from_disk")
    def test_file_not_found_exits_with_error(
        self,
        mock_load: MagicMock,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that FileNotFoundError exits with error."""
        mock_ollama.return_value = None
        mock_load.side_effect = FileNotFoundError("Directory not found")

        with patch("sys.argv", ["embed_and_store.py", "--author", TEST_AUTHOR]):
            from scripts.embed_and_store import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("scripts.embed_and_store.check_ollama_or_exit")
    def test_ollama_not_available_exits_with_error(
        self,
        mock_ollama: MagicMock,
    ) -> None:
        """Test that Ollama not available exits with error."""
        # check_ollama_or_exit handles the error internally and calls sys.exit(1)
        mock_ollama.side_effect = SystemExit(1)

        with patch("sys.argv", ["embed_and_store.py", "--author", TEST_AUTHOR]):
            from scripts.embed_and_store import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    @patch("scripts.embed_and_store.check_ollama_or_exit")
    @patch("scripts.embed_and_store.embed_and_store")
    @patch("scripts.embed_and_store.chunk_documents")
    @patch("scripts.embed_and_store.load_documents_from_disk")
    def test_embedding_exception_exits_with_error(
        self,
        mock_load: MagicMock,
        mock_chunk: MagicMock,
        mock_embed: MagicMock,
        mock_ollama: MagicMock,
        sample_documents: list[Document],
        sample_chunks: list[Document],
    ) -> None:
        """Test that embedding exception exits with error."""
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks
        mock_ollama.return_value = None
        mock_embed.side_effect = Exception("Embedding failed")

        with patch("sys.argv", ["embed_and_store.py", "--author", TEST_AUTHOR]):
            from scripts.embed_and_store import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


class TestEmbedAuthor:
    """Test embed_author() helper function."""

    @patch("scripts.embed_and_store.embed_and_store")
    @patch("scripts.embed_and_store.chunk_documents")
    @patch("scripts.embed_and_store.load_documents_from_disk")
    def test_returns_chunk_count(
        self,
        mock_load: MagicMock,
        mock_chunk: MagicMock,
        mock_embed: MagicMock,
        sample_documents: list[Document],
        sample_chunks: list[Document],
    ) -> None:
        """Test that embed_author returns number of chunks stored."""
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks

        from scripts.embed_and_store import embed_author
        num_chunks = embed_author(TEST_AUTHOR, TEST_INPUT_PATH)

        assert num_chunks == len(sample_chunks)

    @patch("scripts.embed_and_store.load_documents_from_disk")
    def test_invalid_author_raises_value_error(
        self,
        mock_load: MagicMock,
    ) -> None:
        """Test that invalid author raises ValueError."""
        from scripts.embed_and_store import embed_author

        with pytest.raises(ValueError, match="Unknown author"):
            embed_author(INVALID_AUTHOR, TEST_INPUT_PATH)

        # Load should not be called for invalid author
        mock_load.assert_not_called()

    @patch("scripts.embed_and_store.embed_and_store")
    @patch("scripts.embed_and_store.chunk_documents")
    @patch("scripts.embed_and_store.load_documents_from_disk")
    def test_correct_paths_constructed(
        self,
        mock_load: MagicMock,
        mock_chunk: MagicMock,
        mock_embed: MagicMock,
        sample_documents: list[Document],
        sample_chunks: list[Document],
    ) -> None:
        """Test that correct paths are constructed for document loading."""
        mock_load.return_value = sample_documents
        mock_chunk.return_value = sample_chunks

        custom_input = "custom/input"

        from scripts.embed_and_store import embed_author
        embed_author(TEST_AUTHOR, custom_input)

        # Verify correct input path
        expected_input = Path(custom_input) / TEST_DOCUMENT_ID
        mock_load.assert_called_once_with(expected_input)
