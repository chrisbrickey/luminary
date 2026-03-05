"""Unit tests for IO utilities."""

import json
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.utils.io import load_documents_from_disk, save_documents_to_disk


class TestSaveDocumentsToDisk:
    """Test save_documents_to_disk function."""

    def test_saves_documents_to_json_files(self) -> None:
        """Test that documents are correctly saved as JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create sample documents
            docs = [
                Document(
                    page_content="Content of page 1",
                    metadata={"page_number": 1, "author": "test"}
                ),
                Document(
                    page_content="Content of page 2",
                    metadata={"page_number": 2, "author": "test"}
                ),
            ]

            # Save documents
            saved_paths = save_documents_to_disk(docs, tmppath)

            assert len(saved_paths) == 2
            assert all(p.exists() for p in saved_paths)
            assert (tmppath / "page_01.json").exists()
            assert (tmppath / "page_02.json").exists()

            # Verify content
            with open(tmppath / "page_01.json") as f:
                data = json.load(f)
                assert data["page_content"] == "Content of page 1"
                assert data["metadata"]["page_number"] == 1
                assert data["metadata"]["author"] == "test"

    def test_creates_directory_if_not_exists(self) -> None:
        """Test that directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir) / "new" / "nested" / "dir"
            assert not tmppath.exists()

            doc = Document(
                page_content="Test content",
                metadata={"page_number": 1}
            )

            save_documents_to_disk([doc], tmppath)

            assert tmppath.exists()
            assert (tmppath / "page_01.json").exists()

    def test_raises_error_for_missing_page_number(self) -> None:
        """Test that error is raised if page_number is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            doc = Document(
                page_content="Test content",
                metadata={"author": "test"}  # Missing page_number
            )

            with pytest.raises(ValueError) as exc_info:
                save_documents_to_disk([doc], tmpdir)

            assert "missing 'page_number'" in str(exc_info.value)

    def test_handles_empty_document_list(self) -> None:
        """Test that empty document list is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = save_documents_to_disk([], tmpdir)
            assert saved_paths == []

    def test_formats_page_numbers_with_padding(self) -> None:
        """Test that page numbers are zero-padded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            docs = [
                Document(
                    page_content="Page 5",
                    metadata={"page_number": 5}
                ),
                Document(
                    page_content="Page 10",
                    metadata={"page_number": 10}
                ),
            ]

            save_documents_to_disk(docs, tmppath)

            assert (tmppath / "page_05.json").exists()
            assert (tmppath / "page_10.json").exists()

    def test_preserves_unicode_content(self) -> None:
        """Test that Unicode content is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            doc = Document(
                page_content="Français: « liberté, égalité »",
                metadata={"page_number": 1, "title": "Les Lumières"}
            )

            save_documents_to_disk([doc], tmppath)

            with open(tmppath / "page_01.json", encoding="utf-8") as f:
                data = json.load(f)
                assert "Français" in data["page_content"]
                assert "« liberté, égalité »" in data["page_content"]
                assert data["metadata"]["title"] == "Les Lumières"


class TestLoadDocumentsFromDisk:
    """Test load_documents_from_disk function."""

    def test_loads_documents_from_json_files(self) -> None:
        """Test that documents are correctly loaded from JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create sample JSON files
            docs = [
                Document(
                    page_content="Content of page 1",
                    metadata={"page_number": 1, "author": "test-author"}
                ),
                Document(
                    page_content="Content of page 2",
                    metadata={"page_number": 2, "author": "test-author"}
                ),
            ]
            save_documents_to_disk(docs, tmppath)

            # Load documents
            loaded_docs = load_documents_from_disk(tmppath)

            assert len(loaded_docs) == 2
            assert loaded_docs[0].page_content == "Content of page 1"
            assert loaded_docs[0].metadata["page_number"] == 1
            assert loaded_docs[0].metadata["author"] == "test-author"
            assert loaded_docs[1].page_content == "Content of page 2"
            assert loaded_docs[1].metadata["page_number"] == 2

    def test_loads_documents_in_sorted_order(self) -> None:
        """Test that documents are loaded in sorted filename order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Save documents in non-sequential order
            docs = [
                Document(
                    page_content="Page 10",
                    metadata={"page_number": 10}
                ),
                Document(
                    page_content="Page 2",
                    metadata={"page_number": 2}
                ),
                Document(
                    page_content="Page 5",
                    metadata={"page_number": 5}
                ),
            ]
            save_documents_to_disk(docs, tmppath)

            # Load and verify order
            loaded_docs = load_documents_from_disk(tmppath)

            assert len(loaded_docs) == 3
            assert loaded_docs[0].page_content == "Page 2"
            assert loaded_docs[1].page_content == "Page 5"
            assert loaded_docs[2].page_content == "Page 10"

    def test_raises_error_if_directory_not_found(self) -> None:
        """Test that FileNotFoundError is raised if directory doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_documents_from_disk("/nonexistent/directory")

        assert "Directory not found" in str(exc_info.value)

    def test_returns_empty_list_for_directory_with_no_json_files(self) -> None:
        """Test that empty list is returned if no page_*.json files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create some non-matching files
            (tmppath / "readme.txt").write_text("test")
            (tmppath / "data.json").write_text("{}")

            loaded_docs = load_documents_from_disk(tmppath)

            assert loaded_docs == []

    def test_preserves_unicode_content(self) -> None:
        """Test that Unicode content is preserved during load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            doc = Document(
                page_content="Français: « liberté, égalité »",
                metadata={"page_number": 1, "title": "Les Lumières"}
            )
            save_documents_to_disk([doc], tmppath)

            # Load document
            loaded_docs = load_documents_from_disk(tmppath)

            assert len(loaded_docs) == 1
            assert "Français" in loaded_docs[0].page_content
            assert "« liberté, égalité »" in loaded_docs[0].page_content
            assert loaded_docs[0].metadata["title"] == "Les Lumières"

    def test_preserves_all_metadata_fields(self) -> None:
        """Test that all metadata fields are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            doc = Document(
                page_content="Test content",
                metadata={
                    "page_number": 1,
                    "document_id": "test-doc",
                    "document_title": "Test Document",
                    "author": "test-author",
                    "source": "https://example.com",
                    "custom_field": "custom_value",
                }
            )
            save_documents_to_disk([doc], tmppath)

            # Load and verify all fields preserved
            loaded_docs = load_documents_from_disk(tmppath)

            assert len(loaded_docs) == 1
            metadata = loaded_docs[0].metadata
            assert metadata["page_number"] == 1
            assert metadata["document_id"] == "test-doc"
            assert metadata["document_title"] == "Test Document"
            assert metadata["author"] == "test-author"
            assert metadata["source"] == "https://example.com"
            assert metadata["custom_field"] == "custom_value"

    def test_round_trip_save_and_load(self) -> None:
        """Test that documents can be saved and loaded without data loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            original_docs = [
                Document(
                    page_content="First document with some content",
                    metadata={
                        "page_number": 1,
                        "author": "test-author",
                        "title": "Test Title"
                    }
                ),
                Document(
                    page_content="Second document with different content",
                    metadata={
                        "page_number": 2,
                        "author": "test-author",
                        "title": "Test Title"
                    }
                ),
            ]

            # Save and load
            save_documents_to_disk(original_docs, tmppath)
            loaded_docs = load_documents_from_disk(tmppath)

            # Verify complete equality
            assert len(loaded_docs) == len(original_docs)
            for original, loaded in zip(original_docs, loaded_docs):
                assert loaded.page_content == original.page_content
                assert loaded.metadata == original.metadata
